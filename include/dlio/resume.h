/***********************************************************
 *  slamlab: starting DLIO in the MIDDLE of a run.
 ***********************************************************/

#ifndef DLIO_RESUME_H
#define DLIO_RESUME_H

// WHY THIS EXISTS
// ---------------
// Stock DLIO can only start at rest at the beginning of a bag. Three things are
// derived in its first three seconds and then never revisited:
//
//   * the initial attitude   (gravity alignment; the WORLD frame's z axis)
//   * the accelerometer bias
//   * the gyroscope bias
//
// and one more is simply assumed: the rig is STANDING STILL, so position and
// velocity are zero.
//
// A run that starts mid-bag gets all four wrong, and each is wrong in a way that
// does not announce itself:
//
//   * `dlio/imu/calibration:false` is the only switch that accepts prior biases,
//     and it takes the gravity alignment away with them -- leaving state.q at
//     IDENTITY, i.e. a world frame tilted by whatever roll/pitch the rig happened
//     to hold at the pickup instant. propagateState subtracts gravity from world
//     z ONLY, so a tilt of theta injects a permanent horizontal acceleration of
//     g*sin(theta) -- 0.171 m/s^2 per degree -- and the observer's only sink for
//     it is the accel-bias state, which is the exact ratchet that ran the big
//     Sandland bag to |v| = 19.8 m/s.
//   * `dlio/odom/imu/calibration/{accel,gyro}:false` does NOT skip the 3 s
//     window and does NOT enable the priors; with the default
//     `dlio/imu/calibration:true` it yields ZERO biases. It is a trap, not a
//     feature.
//   * calibrating at a moving pickup averages a walking rig: the gyro "bias"
//     becomes the mean turn rate over 3 s (a 30 deg turn writes 0.17 rad/s,
//     40x the real bias) and the estimated up-direction tilts by |a_lateral|/g.
//
// So a resume has to be TOLD, and it has to be told loudly enough that a
// parameter which never arrived cannot look like one that did. This header is
// the arithmetic of being told; odom.cc is the plumbing.
//
// Header-only, pure Eigen, no ROS and no PCL: it is compiled and RUN by the
// image build (test_resume.cpp), so a mistake here turns the BUILD red instead
// of a 40-minute bag replay.
//
// FRAMES, once, because every one of these has been got wrong at least once in
// this codebase:
//
//   q  : WORLD <- BASELINK rotation, the quaternion DLIO stores in state.q and
//        the exporter writes to poses.ndjson as `rotation` (w,x,y,z).
//   p  : position of BASELINK in WORLD, metres (poses.ndjson `translation`).
//   v  : linear velocity in WORLD, m/s. poses.ndjson `twist.linear` is
//        state.v.lin.w VERBATIM -- it is WORLD, not the body frame the ROS
//        Odometry message convention would imply. v_body = R(q)^T * v_world.
//   b  : IMU biases in the BASELINK frame, AFTER transformImu() has rotated the
//        raw IMU by extrinsics.baselink2imu.R and removed the lever arm. They
//        are therefore only valid for a run launched with the SAME extrinsics.
//        accel m/s^2, gyro rad/s.

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dlio {
namespace resume {

//: A quaternion whose norm is further than this from 1 is not a rotation, it is
//: a typo (or three of four fields that arrived and one that did not).
//:
//: THIS IS CHECKED ON THE NUMBERS THAT ARRIVED, NOT ON THE SEED. make_seed()
//: normalises, so by the time verify() sees a Seed the norm is 1 by
//: construction and asking it again answers nothing. A component that arrived
//: as zero normalises into a perfectly valid rotation pointing somewhere else
//: -- the parent's own seed with q.z dropped comes out 106.6 deg away -- so the
//: input norm is RECORDED at construction (Seed::q_norm_in) and that is what is
//: judged. Found by the adversarial review, 2026-09-08: the old check could not
//: fire on the production path, and its test built the Seed by hand to reach it.
static const double QUAT_NORM_TOL = 1e-3;

//: How far an applied prior may sit from the requested one before the run is
//: refused. The values travel as float parameters through the ROS param server
//: and land in a float state, so the tolerance is a float epsilon and not zero:
//: an exact compare would fail on a value that round-tripped perfectly.
static const double APPLIED_TOL = 1e-6;

struct Seed {
  bool have_attitude = false;
  bool have_position = false;
  bool have_velocity = false;
  Eigen::Quaternionf q = Eigen::Quaternionf(1.f, 0.f, 0.f, 0.f);  // WORLD<-BASELINK
  Eigen::Vector3f p = Eigen::Vector3f::Zero();                    // WORLD, m
  Eigen::Vector3f v_w = Eigen::Vector3f::Zero();                  // WORLD, m/s

  //: The norm of the attitude AS IT ARRIVED, before make_seed() normalised it.
  //: 1 when no attitude was seeded, so a seed without one is never complained
  //: about. This is the only surviving evidence that four numbers reached the
  //: node rather than three; normalisation destroys the rest.
  float q_norm_in = 1.f;

  //: What was actually seeded, for the log. Never "resume" on its own: a run
  //: that seeded nothing but claimed to be resuming is the failure this whole
  //: header exists to make impossible.
  std::string describe() const {
    std::string s;
    if (this->have_attitude) s += "attitude";
    if (this->have_position) s += (s.empty() ? "" : "+") + std::string("position");
    if (this->have_velocity) s += (s.empty() ? "" : "+") + std::string("velocity");
    return s.empty() ? std::string("none") : s;
  }
};

//: One thing the run asked for that the run did not get. `key` is the ROS
//: parameter name so the complaint can be pasted straight into a rosparam get.
struct Mismatch {
  std::string key;
  double requested = 0.;
  double effective = 0.;
  std::string note;
};

//: v_body = R(q)^T v_world. The one line of frame arithmetic a resume needs, in
//: one place, so it can be tested rather than inspected.
inline Eigen::Vector3f body_velocity(const Eigen::Quaternionf& q,
                                     const Eigen::Vector3f& v_world) {
  return q.toRotationMatrix().transpose() * v_world;
}

//: Is this quaternion usable as an attitude? A vector of four numbers is not a
//: rotation until it is one.
inline bool quaternion_is_sane(const Eigen::Quaternionf& q, double tol = QUAT_NORM_TOL) {
  const double n = q.norm();
  return std::isfinite(n) && std::fabs(n - 1.0) <= tol;
}

//: The angle (radians) between the world z axis this attitude implies and true
//: up. With a gravity-aligned parent seed this is the rig's own roll/pitch, and
//: it is the number that costs g*sin(theta) of injected horizontal acceleration
//: for the whole child run -- so it is REPORTED, never silently accepted.
//:
//: R(q) maps BASELINK -> WORLD, so R^T * z_world is gravity's direction seen in
//: the body, and the tilt is the angle between the body's own z and that.
inline double tilt_from_gravity_rad(const Eigen::Quaternionf& q) {
  const Eigen::Vector3f up_in_body =
      q.toRotationMatrix().transpose() * Eigen::Vector3f(0.f, 0.f, 1.f);
  const double c = std::max(-1.0, std::min(1.0, (double)up_in_body.z()));
  return std::acos(c);
}

//: Build the seed from what the parameters carried. `have_*` is the caller's
//: statement that a parameter was actually present -- NOT "the value differs
//: from the default", which cannot tell a deliberate zero from an absent key.
inline Seed make_seed(bool have_attitude, const Eigen::Quaternionf& q,
                      bool have_position, const Eigen::Vector3f& p,
                      bool have_velocity, const Eigen::Vector3f& v_w) {
  Seed s;
  s.have_attitude = have_attitude;
  s.have_position = have_position;
  s.have_velocity = have_velocity;
  if (have_attitude) {
    s.q_norm_in = q.norm();          // BEFORE normalising: see QUAT_NORM_TOL
    s.q = q.normalized();
  }
  if (have_position) s.p = p;
  if (have_velocity) s.v_w = v_w;
  return s;
}

//: SILENT NO-OP LAW, the resume instance.
//:
//: Every failure this checks for looks EXACTLY like a healthy run from the
//: outside: the node starts, prints a banner, produces poses. The values it was
//: told to use simply are not the values it is using, because a launch-file
//: <param> tag beat the per-run override yaml (N58), or the yaml never rendered,
//: or the override named `dlio/odom/imu/intrinsics/...` instead of
//: `dlio/imu/intrinsics/...`, or the calibration switch was the one that does
//: nothing. So the run states what it asked for, reads back what it got, and
//: REFUSES when they differ. An empty return is the only clean start.
//:
//: `requested_*` are what the dispatcher put in the override yaml;
//: `effective_*` are read out of the node's own state AFTER getParams().
inline std::vector<Mismatch> verify(bool requested_calibration_off,
                                    bool effective_calibration_off,
                                    const Eigen::Vector3f& requested_accel_bias,
                                    const Eigen::Vector3f& effective_accel_bias,
                                    const Eigen::Vector3f& requested_gyro_bias,
                                    const Eigen::Vector3f& effective_gyro_bias,
                                    const Eigen::Matrix3f& effective_accel_sm,
                                    const Seed& seed,
                                    double tol = APPLIED_TOL) {
  std::vector<Mismatch> out;

  if (requested_calibration_off && !effective_calibration_off) {
    Mismatch m;
    m.key = "dlio/imu/calibration";
    m.requested = 0.;
    m.effective = 1.;
    m.note = "the run asked for frozen priors, and the node is still going to "
             "spend 3 s calibrating a MOVING rig instead";
    out.push_back(m);
  }

  // The biases are only meaningful when calibration is off: with it on,
  // getParams DISCARDS the priors (odom.cc: the else branch zeroes state.b),
  // so comparing them would report a second, derived complaint for one cause.
  if (requested_calibration_off && effective_calibration_off) {
    static const char* axis[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
      if (std::fabs(requested_accel_bias[i] - effective_accel_bias[i]) > tol) {
        Mismatch m;
        m.key = std::string("dlio/imu/intrinsics/accel/bias[") + axis[i] + "]";
        m.requested = requested_accel_bias[i];
        m.effective = effective_accel_bias[i];
        m.note = "m/s^2, BASELINK frame";
        out.push_back(m);
      }
      if (std::fabs(requested_gyro_bias[i] - effective_gyro_bias[i]) > tol) {
        Mismatch m;
        m.key = std::string("dlio/imu/intrinsics/gyro/bias[") + axis[i] + "]";
        m.requested = requested_gyro_bias[i];
        m.effective = effective_gyro_bias[i];
        m.note = "rad/s, BASELINK frame";
        out.push_back(m);
      }
    }

    // TRAP A. The scale-misalignment matrix is gated on the SAME flag as the
    // priors: with calibration on it is forced to identity and the parameter is
    // silently discarded, so a resume run is the first run that ever APPLIES it.
    // If it is not identity, the child's corrected acceleration sm*a - b is not
    // the same function of the raw IMU that the parent used, and the frozen bias
    // is being spent in a frame it was never measured in.
    const double sm_err = (effective_accel_sm - Eigen::Matrix3f::Identity())
                              .cwiseAbs().maxCoeff();
    if (sm_err > tol) {
      Mismatch m;
      m.key = "dlio/imu/intrinsics/accel/sm";
      m.requested = 0.;
      m.effective = sm_err;
      m.note = "must stay IDENTITY on a resume: the parent measured its biases "
               "with sm forced to identity, so applying one now changes what "
               "the frozen bias means";
      out.push_back(m);
    }
  }

  //: Judged on the arriving norm, not the normalised seed: see QUAT_NORM_TOL.
  const bool q_arrived_sane =
      std::isfinite((double)seed.q_norm_in) &&
      std::fabs((double)seed.q_norm_in - 1.0) <= QUAT_NORM_TOL &&
      quaternion_is_sane(seed.q);

  if (seed.have_attitude && !q_arrived_sane) {
    Mismatch m;
    m.key = "dlio/resume/initial/attitude";
    m.requested = 1.;
    m.effective = seed.q_norm_in;
    m.note = "not a unit quaternion as it ARRIVED (w,x,y,z, WORLD<-BASELINK): "
             "normalising it would produce a valid rotation pointing somewhere "
             "the parent never was";
    out.push_back(m);
  }

  // A seed that is exactly identity while a seed was requested is a parsed
  // no-op: four numbers arrived, none of them said anything. The parent's own
  // attitude at any pickup worth resuming from is never exactly identity --
  // gravity alignment alone puts roll/pitch a degree or two off.
  if (seed.have_attitude && q_arrived_sane) {
    const double off = std::fabs(1.0 - std::fabs((double)seed.q.w()));
    if (off < 1e-9) {
      Mismatch m;
      m.key = "dlio/resume/initial/attitude";
      m.requested = 0.;
      m.effective = 0.;
      m.note = "IDENTITY was seeded: the parameter parsed but says nothing, "
               "which is the shape of an override that never reached the node";
      out.push_back(m);
    }
  }

  return out;
}

}  // namespace resume
}  // namespace dlio

#endif  // DLIO_RESUME_H
