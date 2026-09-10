#ifndef DLIO_SMOOTHER_H
#define DLIO_SMOOTHER_H

// =========================================================================
//  E4 / ARCHITECT A -- A FIXED-LAG SMOOTHER BEHIND DLIO'S OWN REGISTRATION
//
//  keeper/lio_concept/dlio_smoother/A_nudge/DESIGN.md   (the architecture)
//  keeper/lio_concept/dlio_smoother/E0/E0.md            (the offline proof)
//  keeper/lio_concept/dlio_smoother/E4/E2/E2.md         (the true 6x6)
//  keeper/lio_concept/dlio_smoother/E4/IMPL/IMPL.md     (as built)
//
//  WHAT THIS IS. DLIO's observer has no covariance anywhere: updateState()
//  pulls the state toward the GICP pose at a fixed Kp/Kv/Kq whether or not
//  the geometry supported that pose. A fixed-lag smoother can say "I do not
//  know yet" and REPLACE the weak-axis increment with the posterior of the
//  IMU chain over the window -- which is a better estimate than either input
//  and needs no threshold to fire. E0 measured that on recorded data: where
//  the input read 52.89 deg and 40.51 deg against its own gyro through the
//  bend, every smoothed arm read 1.35-2.10 deg, on 3 of 3 draws, at 4 of 4
//  lags, under both information bounds.
//
//  THERE IS NO DEGENERACY BAND IN THIS FILE. Not a min_ratio, not a
//  min_weight, not a max_weak_dirs, not a min_info. Four hand-fitted bands
//  have died in this programme, in BOTH directions, and all four WITHHELD
//  correction -- which leaves the weak axis on raw IMU integration with no
//  other opinion. The smoother does the opposite: the arbitration is the
//  ratio of two covariances and it is continuous. The only constants here
//  are (a) one scale that converts a correspondence COUNT into an
//  information in m^-2, fitted once offline on healthy data and never per
//  bag (E0 sec 3), and (b) a deliberately loose prior added to every
//  registration factor so a rank-deficient block stays well posed. Neither
//  is a threshold: (a) multiplies, (b) is a constant that is inert wherever
//  the geometry speaks.
//
//  THE SILENT NO-OP THIS DESIGN DIES OF. A correction written into
//  state.{p,q,v} and NOTHING ELSE survives exactly one scan: the next
//  deskewPointcloud() re-anchors the prior on the untouched lidarPose
//  (odom.cc: integrateImu(prev_scan_stamp, lidarPose.q, lidarPose.p,
//  geo.prev_vel, ...)) and the next align() measures against a submap the
//  smoother did not move. So the write-back is SIX groups at one line and
//  write_back() below counts them: a report that does not say 6 is a bug,
//  and test_smoother.cpp turns red for each one that is skipped.
//      1  T / T_corr        the pose AND the published cloud (they ride the
//                           same matrix, odom.cc publishCloud)
//      2  lidarPose         the next scan's GICP prior anchor
//      3  state.p/q/v       the observer's own state
//      4  geo.prev_*        the next prior's velocity
//      5  state.b           the biases (Kab/Kgb may then be set to 0)
//      6  in-window keyframe poses + the submap dirty flag
//
//  FRAMES, stated once because getting one wrong is silent.
//    * X(k) is `world <- baselink`, i.e. exactly DLIO's `T`.
//    * V(k) is the WORLD-frame linear velocity, i.e. state.v.lin.w.
//    * B(k) is the bias in the IMU SENSOR frame, because that is what
//      gtsam's CombinedImuFactor subtracts from the raw sample before
//      body_P_sensor is applied. DLIO's state.b is in BASELINK. The two are
//      related by extrinsics.baselink2imu.R and the conversion is in one
//      place (bias_bl_from_sensor / bias_sensor_from_bl) with a fixture.
//    * The IMU fed to the smoother is the RAW sensor-frame stream, tapped in
//      callbackImu BEFORE transformImu(). DLIO's own imu_buffer is
//      bias-corrected at arrival with whatever bias was current (odom.cc
//      callbackImu), so it cannot be reused by an estimator that re-estimates
//      bias; and transformImu() differentiates omega at 640 Hz to build a
//      lever-arm term, which gtsam's body_P_sensor does analytically.
//    * The registration information H6b arrives already SENSOR-ANCHORED in
//      gtsam::Pose3's RIGHT/BODY tangent, [rot(0..2) | trans(3..5)] --
//      dlio/degeneracy.h body_information(), E2's own conversion, unit-tested
//      there at 1e-9 against the re-anchoring identity.
// =========================================================================

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "dlio/degeneracy.h"   // skew3, log_so3, upper21/from_upper21, adjoint

// Sabotage switch for the build-time fixtures. 0 = the shipping code. Every
// non-zero value BREAKS one specific write, and test_smoother.cpp must go red
// for each -- a fixture that cannot fail is decoration ([[silent_no_op_law]]).
//   1  the lidarPose write is skipped   (the nudge never reaches the prior)
//   2  the T_corr write is skipped      (pose and cloud disagree)
//   3  the keyframe deltas and the submap dirty flag are skipped
//   4  alpha == 0 writes anyway         (the A/A control is not a control)
//   5  the cross block of H6b is dropped (E0's forced reconstruction)
#ifndef DLIO_SMOOTHER_SABOTAGE
#define DLIO_SMOOTHER_SABOTAGE 0
#endif

namespace dlio {
namespace smoother {

// ------------------------------------------------------------------ math ---

// SO(3) exponential, the inverse of dlio::degeneracy::log_so3. Rodrigues with
// the small-angle series taken below 1e-8 rad, where sin(t)/t loses digits.
inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
  const double th = w.norm();
  if (th < 1e-10) {
    // second order is enough: the neglected term is O(th^3) < 1e-30
    return Eigen::Matrix3d::Identity() + degeneracy::skew3(w)
           + 0.5 * degeneracy::skew3(w) * degeneracy::skew3(w);
  }
  const Eigen::Matrix3d K = degeneracy::skew3(w / th);
  return Eigen::Matrix3d::Identity() + std::sin(th) * K
         + (1.0 - std::cos(th)) * K * K;
}

// The DECOUPLED geodesic blend A -> B at fraction alpha, in the SAME chart
// every 6-vector in dlio/degeneracy.h uses:
//     v = (log_so3(R_A^T R_B), R_A^T (t_B - t_A))
//     out = ( R_A exp_so3(alpha v_rot), t_A + alpha R_A v_trans )
// alpha = 0 returns A EXACTLY (R_A * I and t_A + 0) and alpha = 1 returns B
// exactly; both endpoints are asserted bitwise in test_smoother.cpp. The SE(3)
// exponential would also do, and its V(omega) machinery would buy nothing here:
// this is a blend along a path, not a chart the information lives in.
inline Eigen::Matrix4d blend_pose(const Eigen::Matrix4d& A,
                                  const Eigen::Matrix4d& B, double alpha) {
  if (alpha == 0.0) return A;
  if (alpha == 1.0) return B;
  const Eigen::Matrix3d RA = A.block<3, 3>(0, 0);
  const Eigen::Vector3d tA = A.block<3, 1>(0, 3);
  const Eigen::Matrix<double, 6, 1> v = degeneracy::relative_rot_trans(
      RA, tA, B.block<3, 3>(0, 0), B.block<3, 1>(0, 3));
  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out.block<3, 3>(0, 0) = RA * exp_so3(alpha * v.head<3>());
  out.block<3, 1>(0, 3) = tA + alpha * (RA * v.tail<3>());
  return out;
}

// |translation| in metres and |rotation| in degrees between two poses.
inline void pose_difference(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B,
                            double* d_m, double* d_deg) {
  const Eigen::Matrix<double, 6, 1> v = degeneracy::relative_rot_trans(
      A.block<3, 3>(0, 0), A.block<3, 1>(0, 3),
      B.block<3, 3>(0, 0), B.block<3, 1>(0, 3));
  if (d_m) *d_m = v.tail<3>().norm();
  if (d_deg) *d_deg = v.head<3>().norm() * 180.0 / M_PI;
}

// ---------------------------------------------------------------- params ---

struct Params {
  // --- the switch. FALSE IN THE IMAGE: the recipe row turns it on, so the
  // product path is untouched by the presence of this code. ---------------
  bool   enabled = false;

  double lag_s = 5.0;        // dlio/smoother/lag_s   -- WINDOW OF SCANS
  double alpha = 0.5;        // dlio/smoother/alpha   -- 0 = run, log, write nothing

  // The ONE fitted number, twice. E0 fitted s_t = 0.0315 m^-2 from the design's
  // own sanity anchor (implied sigma 3.0 mm on the best-observed chamber axis,
  // which reproduces B_replace sec 6.3's independently derived 0.060 m^-2 at its
  // own lambda to 0.1 %). E2 sec 8 shows the adjoint leaves the translation
  // block's SPECTRUM invariant (Lambda_tt = R^T H_tt R), so that anchor carries
  // across to the true matrix unchanged -- and that the SAME scalar is the right
  // model for the attitude half once the block is sensor-anchored (the two
  // independently fitted scales agree to 1.8 % on Dave, whose lever arm is
  // small). They are two keys so the joint-NIS confirmation E2 still owes can
  // move one without the other, NOT so anybody can tune them per bag.
  // KILL, from A_nudge sec 0.8: if this has to be re-fitted per bag it is the
  // fifth hand-fitted band and the lane stops.
  // E0's own fitted value to all its digits (out/scale_fit.json s_t), not the
  // rounded 0.0315 the reports quote -- the offline A/A compares trajectories,
  // and a rounded scale would show up there as a difference this code caused.
  double info_scale_trans = 0.031503901758232755;   // m^-2 per correspondence count
  double info_scale_rot   = 0.031503901758232755;   // rad^-2 per (count * m^2)

  double huber_k = 1.345;             // CONVEX, never redescending

  // The loose prior ADDED to every registration information. E0 clamped the
  // smallest eigenvalue of each 3x3; a clamp cannot be applied blockwise to a
  // matrix with a cross block without breaking PSD, so this ADDS the same prior
  // as information, which is what a prior IS. Constant, PSD-safe, applied
  // identically to every scan, and inert wherever the geometry speaks (the
  // chamber's own translation information is ~3e4 against a floor of 1e-2).
  double floor_sigma_trans_m  = 10.0;
  double floor_sigma_rot_deg  = 30.0;

  // --- the rig's OWN measured Allan analysis, PER AXIS, SENSOR frame -------
  // reeval/check3_noise.json "rig2860 Ouster ICM (TODAY bigbag)". Never a
  // datasheet, and never GLIM's shipped imu_gyro_noise 0.02, which is 39x the
  // truth. The worst axis is 3.9x the best in gyro and 5.3x in accel, so the
  // isotropic-worst form is 15x pessimistic on two of three axes (E0 sec 9.1;
  // its `iso` rung moved the bend by 0.16 deg and is carried as a sensitivity
  // arm, not as the default).
  double vrw[3]   = {0.007539,  0.002254,  0.001412};    // m/s/sqrt(s)
  double arw[3]   = {0.0001769, 0.0005962, 0.0001543};   // rad/sqrt(s)
  // The bias random walk is DERIVED, not transcribed: the same analysis reports
  // an Allan DEVIATION at tau = 2 s and the PSD follows from ADEV =
  // sigma_rw * sqrt(tau/3), i.e. sigma_rw = ADEV / sqrt(2/3). One authority for
  // the arithmetic (e0_lib.py SIG_BA_RW3 / SIG_BG_RW3 use the same line), so a
  // hand-copied derived number cannot drift away from the measurement.
  double acc_adev_2s[3] = {0.00209,  0.00128,   0.00067};   // m/s^2  at tau = 2 s
  double gyr_adev_2s[3] = {5.31e-05, 1.244e-04, 5.06e-05};  // rad/s  at tau = 2 s
  double integration_sigma = 1e-5;    // integration covariance = this^2 = 1e-10
  double gravity = 9.80665;           // MakeSharedU: n_gravity = (0,0,-g)

  // --- the gravity / attitude anchoring, A_nudge sec 3.3 F5 ----------------
  // DLIO's world frame is gravity-aligned by callbackImu's own init and the
  // accelerometer constrains roll and pitch inside EVERY CombinedImuFactor. The
  // anchor is therefore the gauge prior on X(0): TIGHT in roll/pitch (gravity is
  // the witness), loose but non-singular in yaw and x/y/z. A_nudge names a
  // SECOND gravity opinion inside the window as "an unforced risk" and
  // deliberately does not build it; E0's graph is this one, so reproducing E0
  // requires this one. Values are E0's PREREG sec 2, unchanged.
  double gauge_sigma_rp_deg  = 0.5;
  double gauge_sigma_yaw_deg = 10.0;
  double gauge_sigma_pos_m   = 10.0;

  // --- E4 INCREMENT 2: THE WINDOW IS RE-ANCHORED ON DLIO'S OWN POSE -------
  //
  // WHY THIS EXISTS, measured, not argued (E4/REVIEW.md sec 8.1, PROOF.md
  // sec 7). With the gauge prior above and nothing else, the window's ABSOLUTE
  // placement is free to 10 m and the marginalisation prior carries almost
  // nothing along a corridor's unobservable axis -- so the smoother's estimate
  // of scan k sits at a STANDING FRAME OFFSET from DLIO's own pose for scan k:
  // measured at alpha = 0, where the write-back writes NOTHING and therefore
  // cannot have caused what it is measuring, it stands at 49.2 mm and only
  // 0.33 mm of it is new each scan (lag-10 autocorrelation 0.988). The
  // write-back then blends alpha x THAT into lidarPose, state, geo.prev_* and
  // the map, ten times a second, against a real motion of ~20 mm per scan: a
  // constant bias injected every scan, keyframes at 10x the stock rate, and
  // 1,329 km of trajectory before the OOM killer.
  //
  // THE FIX, and it is the cheap one. Put a TIGHT prior on the OLDEST pose
  // that will SURVIVE this update, at DLIO's own value for that scan. The
  // window's gauge is then DLIO's own frame BY CONSTRUCTION, the standing
  // offset is identically zero, and what remains in `corr` is only the
  // window's own disagreement about the SHAPE of the last lag_s seconds --
  // which is the quantity this design was always supposed to blend. It touches
  // no write: the six writes are unchanged.
  //
  // 1 mm / 0.01 deg is not a fitted band. It is "tight enough that the anchor
  // wins against a marginalisation prior whose weak axis reaches 1.5e6 m^2",
  // and the fixture asserts the offset it removes, not the number itself.
  bool   reanchor = true;
  // HOW the anchor is imposed, and the two are NOT equivalent -- measured:
  //
  //  anchor_in_graph = FALSE (the default, "output"): the window's gauge is
  //    left exactly as E0 built it and the ANSWER is re-expressed in DLIO's
  //    frame -- X_out(k) = T_dlio(k_a) . X(k_a)^-1 . X(k), a rigid gauge
  //    transform applied to the pose, the velocity and the in-window keyframe
  //    poses together. The standing offset is then ZERO BY CONSTRUCTION (the
  //    anchor pose is DLIO's own, exactly), the graph's conditioning is
  //    untouched, and nothing accumulates.
  //
  //  anchor_in_graph = TRUE ("prior"): a tight PriorFactor on that same pose,
  //    at DLIO's own value, replacing the 10 m gauge. This is the literal form
  //    E4/REVIEW.md sec 8.1 names, it is built, and it is NOT the default,
  //    because it does not survive its own closed-loop fixture: the anchor
  //    that is marginalised out leaves its TIGHT absolute information inside
  //    the marginalisation prior, the next scan's anchor sits at a DLIO pose
  //    that the write-back has since moved, and the two tight and mutually
  //    inconsistent absolute constraints fight -- on the synthetic corridor
  //    replay at alpha = 0.5 it reaches 40 m of correction where the gauge
  //    transform reaches millimetres (test_smoother_gtsam.cpp G10/G11). It is
  //    kept, behind this key, because "the obvious form of the fix diverges"
  //    is a result and deleting it would leave nothing to read it off.
  bool   anchor_in_graph = false;
  double anchor_sigma_pos_m   = 0.001;   // 1 mm   (anchor_in_graph only)
  double anchor_sigma_rot_deg = 0.01;    // 0.01 deg
  // THE ABLATION, PROOF.md sec 7 option (a): blend the smoother's own
  // INCREMENT X(k-1)^-1 X(k) onto DLIO's own T_{k-1}^-1 T_k instead of its
  // absolute pose. Frame-offset-free whatever the gauge does, and it is here
  // so the increment can attribute the result to the re-anchoring rather than
  // assert it. Not the default: (b) is, because (b) also repairs the map
  // write-back and the marginal, and (a) only repairs the pose.
  bool   blend_increment = false;
  // At alpha = 0 the smoother's marginal must TRACK DLIO's pose. Scans where
  // it does not, and the registration was present, are counted; the majority
  // of them is the defect above. PROOF.md sec 7's own fixture threshold.
  double standing_offset_max_m = 0.002;

  // --- E4 INCREMENT 3, CANDIDATE A: DO NOT HAND THE BIAS BACK -------------
  //
  // TRUE in the image, so :smoother-2 stays reproducible from this tree; the
  // trial row turns it off. WHY IT EXISTS, measured on Dave cheese at
  // alpha = 0.5 with the frame defect already fixed (E4/INC2/INC2.md sec 8):
  // the write-back injects a slow ROLL/PITCH drift that grows monotonically
  // through the run -- TILT p50 4.22 deg against a 0.35 deg noise floor,
  // 1.71 -> 3.77 -> 4.48 -> 7.76 by quarter, 9.50 deg at the end -- while YAW,
  // the direction gravity cannot see, wanders and returns. It is not the
  // applied rotations adding up: their magnitudes sum to 393.8 deg for a
  // 9.5 deg net effect. What DOES move with it is the accelerometer bias,
  // which walks 64 % further than the alpha = 0 control's and puts the whole
  // of that extra walk in ONE HORIZONTAL AXIS (y: -0.029 -> +0.096, a
  // 0.125 m/s^2 swing = 0.73 deg of equivalent tilt before DLIO's own
  // Kab = 0.25 loop acts on it at all). A horizontal specific-force bias and a
  // tilt of the gravity vector are THE SAME OBSERVATION to the filter, and
  // E4/REVIEW.md sec 8 had already named the shape: two bias mechanisms
  // pulling against each other, DLIO's clamped loop against the graph's
  // unclamped random walk, with the sixth write group handing the graph's
  // answer to the loop ten times a second.
  //
  // Turning it off does NOT stop the smoother estimating the bias: the bias
  // stays in the window, inside the CombinedImuFactor, which is what makes the
  // increments good. It stops it being handed to a filter that already has its
  // own loop for the same quantity.
  bool   writeback_bias = true;
  double v0_sigma            = 0.5;    // m/s
  double bias_prior_sigma_accel = 0.05;   // m/s^2   -- a seed, not an assertion
  double bias_prior_sigma_gyro  = 0.005;  // rad/s
  double bias_init_acc = 0.05;   // BiasAccOmegaInit, accel half
  double bias_init_gyr = 0.005;  // BiasAccOmegaInit, gyro half

  // --- solver -------------------------------------------------------------
  int    lm_max_iterations = 20;
  double lm_relative_error_tol = 1e-8;

  // The linear solver inside the fixed-lag LM. MULTIFRONTAL_CHOLESKY, which is
  // what e0_solve.py and e2_solve.py use -- the offline A/A compares a
  // trajectory, and a different factorisation is a difference this code would
  // have introduced. "MULTIFRONTAL_QR" is reachable and measured beside it
  // (IMPL.md sec 7): it is ~5x slower and, once the information matrices are
  // right, buys nothing. It bought a great deal while they were WRONG, which is
  // how the zero-information bug of IMPL.md sec 6.3 first showed itself --
  // worth remembering the next time a solver change looks like a fix.
  std::string linear_solver = "MULTIFRONTAL_CHOLESKY";

  // --- compute budget lever (IMPL.md sec 7). 1 = solve every scan. N > 1
  // accumulates N scans of factors and calls update() once, which is still a
  // fixed lag; the write-back then applies on the scans that solved. Measured
  // and reported per run either way; this exists so a box that cannot hold the
  // scan period has a lever that is NOT "make the window shorter".
  int    marginalize_every = 1;

  // How often the marginal on X(k) is actually computed. It is a factorisation
  // of the window graph, not a lookup (gtsam 4.2.2's
  // BatchFixedLagSmoother::marginalCovariance THROWS "not implemented"), so it
  // has its own budget lever. 1 = every scan. Setting it above 1 leaves ZEROS
  // on the [SMOOTH] line's mcov field for the scans it skipped -- zeros, which
  // read as "not asked for", never -1, which reads as "asked and failed".
  int    marg_every = 1;

  // --- keyframes ----------------------------------------------------------
  bool   keyframe_writeback = true;
  double kf_dirty_trans_m = 0.005;   // 5 mm, well under the 10-12 mm
  double kf_dirty_rot_deg = 0.05;    // registration floor, so sub-floor
                                     // motion never triggers work

  // A sample interval longer than this is DECLARED rather than integrated
  // through (DLIO has no gap handling at all: a 294 ms gap is integrated as one
  // step at one rate). On the fixed player this is moot -- 640.00 Hz, 0 gaps,
  // asserted per arm -- and the record must show it fired zero times rather than
  // not existing.
  double imu_gap_s = 3.0 / 640.0;

  int    log_every = 1;
};

// ---------------------------------------------------------------- inputs ---

struct ImuSample {
  double stamp = 0.0;
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // RAW, sensor frame, m/s^2
  Eigen::Vector3d gyro  = Eigen::Vector3d::Zero();  // RAW, sensor frame, rad/s
};

struct ScanInput {
  double stamp = 0.0;          // this scan's absolute stamp (scan_stamp)
  double prev_stamp = 0.0;     // the previous scan's, for the IMU window
  Eigen::Matrix4d T_gicp = Eigen::Matrix4d::Identity();   // world <- baselink
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();      // initial value for V(k)

  // The registration's own information, SENSOR-ANCHORED, gtsam [rot|trans]
  // tangent -- degeneracy::body_information(getFinalHessian(), R, t).
  Eigen::Matrix<double, 6, 6> H6b = Eigen::Matrix<double, 6, 6>::Zero();
  bool h_valid = false;
  int  ncorr = 0;
  double ferr = 0.0;

  // The registration's own relative pose T_{k-1}^-1 T_k, DECOUPLED chart.
  Eigen::Matrix<double, 6, 1> dp6 = Eigen::Matrix<double, 6, 1>::Zero();
  bool dp6_valid = false;

  std::vector<ImuSample> imu;   // raw samples in (prev_stamp, stamp]
};

struct Solution {
  bool valid = false;
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();   // smoothed pose, this scan
  Eigen::Vector3d v = Eigen::Vector3d::Zero();       // world velocity
  Eigen::Vector3d b_accel = Eigen::Vector3d::Zero(); // SENSOR frame
  Eigen::Vector3d b_gyro  = Eigen::Vector3d::Zero(); // SENSOR frame

  // in-window keyframes the smoother moved: (keyframe index, new world pose)
  std::vector<std::pair<int, Eigen::Matrix4d>> kf_poses;
  // keyframes whose scan LEFT the window on this update. They are frozen from
  // here on and any later delta offered to them is a hard error, not a skip.
  std::vector<int> kf_frozen;

  double marg_cov[6] = {0, 0, 0, 0, 0, 0};   // diag of the marginal on X(k)
  double resid_imu = 0.0;      // whitened error of THIS scan's IMU factor
  double resid_reg = 0.0;      // whitened error of THIS scan's registration factor
  double solve_ms = 0.0;
  int    window_vars = 0;
  int    window_factors = 0;
  int    lm_iterations = 0;
  int    imu_samples = 0;
  int    imu_gaps = 0;
  // HOW FAR THE IMU STREAM IS AHEAD OF THIS SCAN, in seconds: the newest raw
  // sample the tap holds minus this scan's own stamp. It IS the pipeline
  // backlog, and when it exceeds raw_imu_keep_s the tap runs dry -- so the
  // record shows the horizon and not only the moment it was crossed.
  double imu_lead_s = 0.0;
  bool   floor_binding = false;
  bool   psd_projected = false;
  bool   update_exception = false;
  long   singular_info = 0;    // cumulative: registration factors REFUSED
                               // because their information was singular
  long   reseats = 0;          // cumulative: a fixed-lag window rebuilt from
                               // scratch after a failed update. Bounded and
                               // COUNTED; a run with any is not a clean arm.
  bool   solved_this_scan = false;   // false on the scans marginalize_every skips

  // --- E4 INCREMENT 2 -------------------------------------------------------
  // The window's own relative motion over the LAST SCAN, and DLIO's, in the
  // same chart. Both are gauge-free by construction, which is the whole point:
  // PROOF.md sec 7 option (a) blends their difference instead of the absolute
  // pose, and the [SMOOTH] record carries it either way so a reader can see
  // how much of `corr` was frame and how much was motion.
  Eigen::Matrix4d T_rel_smoother = Eigen::Matrix4d::Identity();  // X(k-1)^-1 X(k)
  Eigen::Matrix4d T_rel_dlio     = Eigen::Matrix4d::Identity();  // T_{k-1}^-1 T_k
  bool   rel_valid = false;
  // The pose the window was re-anchored on this scan, and how far the anchor
  // had to move it. anchor_key < 0 means no anchor was added on this scan
  // (either reanchor is off, or the window's oldest survivor is unchanged).
  long   anchor_key = -1;
  bool   anchor_added = false;
  // the world->world gauge the output re-anchor applied. Identity when the
  // re-anchor is off or in the graph, and the keyframe poses above have
  // ALREADY been carried through it -- pose, velocity and map move together or
  // the gauge is just the old defect wearing a new name.
  Eigen::Matrix4d anchor_gauge = Eigen::Matrix4d::Identity();
  double anchor_resid_m = 0.0;   // |DLIO's pose - the window's own| at the anchor
  double anchor_resid_deg = 0.0;
};

// ------------------------------------------------------- the information ---

// Lambda = s * H6b + FLOOR, in gtsam's [rot(0..2) | trans(3..5)] tangent.
//
// H6b is a correspondence count (PLANE regularisation forces every point
// covariance to (1,1,1e-3), so H_tt = 0.5 N I + 499.5 sum n n^T), NOT an
// information in m^-2; `s` is the scalar that converts it and it is fitted
// ONCE, offline, on healthy data. The rotation half carries count * m^2 for the
// same reason, and E2 proved the adjoint has already removed the world-origin
// lever arm from it, so the SAME scalar is the right model there.
//
// Symmetrised and PSD-projected on the way out: a logged or accumulated matrix
// can be indefinite at round-off and gtsam's Gaussian::Information would take a
// silent LDLT failure downstream. The projection is COUNTED, never silent.
inline Eigen::Matrix<double, 6, 6> information_from_h6b(
    const Eigen::Matrix<double, 6, 6>& H6b, bool h_valid,
    double s_rot, double s_trans, double floor_sigma_rot_deg,
    double floor_sigma_trans_m, bool* psd_projected, bool* floor_binding) {
  Eigen::Matrix<double, 6, 6> L = Eigen::Matrix<double, 6, 6>::Zero();
  if (h_valid && H6b.allFinite()) {
    L.block<3, 3>(0, 0) = s_rot   * H6b.block<3, 3>(0, 0);
    L.block<3, 3>(3, 3) = s_trans * H6b.block<3, 3>(3, 3);
    // THE CROSS BLOCK. It is not decoration: it is the entry E0 could not have
    // and the reason E2 exists. Scaling it by sqrt(s_r s_t) keeps the whole 6x6
    // congruent to a single-scalar model when s_rot == s_trans (which is what
    // ships) and keeps it PSD when they differ.
    const double s_x = std::sqrt(std::max(s_rot, 0.0) * std::max(s_trans, 0.0));
#if DLIO_SMOOTHER_SABOTAGE == 5
    (void)s_x;   // SABOTAGE 5: drop the cross block -- E0's forced reconstruction
#else
    L.block<3, 3>(0, 3) = s_x * H6b.block<3, 3>(0, 3);
    L.block<3, 3>(3, 0) = s_x * H6b.block<3, 3>(3, 0);
#endif
  }
  L = 0.5 * (L + L.transpose()).eval();

  // sigma <= 0 means NO floor. That is not a mode the node ever runs -- it
  // exists so the offline A/A can hand over an information matrix that already
  // carries the same constant, and adding it twice would be a difference this
  // code invented.
  const double fr = floor_sigma_rot_deg > 0.0
                        ? 1.0 / std::pow(floor_sigma_rot_deg * M_PI / 180.0, 2)
                        : 0.0;
  const double ft = floor_sigma_trans_m > 0.0
                        ? 1.0 / std::pow(floor_sigma_trans_m, 2)
                        : 0.0;
  Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Zero();
  F.diagonal() << fr, fr, fr, ft, ft, ft;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(L);
  if (es.info() == Eigen::Success && es.eigenvalues()(0) < 0.0) {
    Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
    for (int i = 0; i < 6; ++i) w(i) = std::max(w(i), 0.0);
    L = es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
    if (psd_projected) *psd_projected = true;
  }
  L += F;
  if (floor_binding) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es2(L);
    *floor_binding = (ft > 0.0) && (es2.info() == Eigen::Success) &&
                     (es2.eigenvalues()(0) <= 2.0 * ft);
  }
  return L;
}

// ------------------------------------------------------------ write-back ---

// A keyframe is a scan, and updateKeyframes() already stores lidarPose, so
// "moving a keyframe" is "reading X(k_j) for a keyframe scan still in the
// window". The sink queues the world->world delta; buildKeyframesAndSubmap()
// applies it to the cloud and the covariances before composing the submap and
// forces the kd-tree rebuild. FREEZE ON MARGINALISATION: a keyframe whose scan
// has left the window is frozen forever and queueDelta() must REFUSE it -- a
// hard error at the call site, never a silent skip. Loop closure lives in the
// owner's layer, so no marginalised pose ever moves again, and that is the one
// property that makes this tractable inside a 100 ms budget.
class KeyframeSink {
 public:
  virtual ~KeyframeSink() = default;
  virtual bool queueDelta(int kf_index, const Eigen::Matrix4d& delta,
                          const Eigen::Matrix4d& pose_new) = 0;
  virtual void markSubmapDirty() = 0;
  virtual bool poseOf(int kf_index, Eigen::Matrix4d* out) const = 0;
};

struct WriteTargets {
  Eigen::Matrix4f*    T = nullptr;
  Eigen::Matrix4f*    T_corr = nullptr;
  const Eigen::Matrix4f* T_prior = nullptr;
  Eigen::Vector3f*    lidar_p = nullptr;
  Eigen::Quaternionf* lidar_q = nullptr;
  Eigen::Vector3f*    state_p = nullptr;
  Eigen::Quaternionf* state_q = nullptr;
  Eigen::Vector3f*    v_world = nullptr;
  Eigen::Vector3f*    v_body = nullptr;
  Eigen::Vector3f*    geo_prev_p = nullptr;
  Eigen::Quaternionf* geo_prev_q = nullptr;
  Eigen::Vector3f*    geo_prev_vel = nullptr;
  Eigen::Vector3f*    b_accel_bl = nullptr;   // BASELINK frame, as DLIO holds it
  Eigen::Vector3f*    b_gyro_bl = nullptr;
  KeyframeSink*       kf = nullptr;
  // The keyframe DIRTY thresholds. Well under the 10-12 mm registration floor,
  // so sub-floor motion never triggers a cloud transform and a kd-tree rebuild.
  // They are not a band: nothing is attenuated and no decision is taken on
  // them; they only say when moving a map is worth the arithmetic.
  double kf_dirty_trans_m = 0.005;
  double kf_dirty_rot_deg = 0.05;
  Eigen::Matrix3f     R_bl_imu = Eigen::Matrix3f::Identity();  // baselink <- imu
  // E4 INCREMENT 3 candidate A. FALSE removes write group 5 and ONLY group 5;
  // the report then EXPECTS five groups instead of six, so a missing sixth is
  // still a hard error and a written fifth is a hard error too. A flag that
  // made "5 of 6" acceptable would have disarmed the law it is standing next
  // to ([[silent_no_op_law]]).
  bool                write_bias = true;
};

struct WriteBackReport {
  bool   applied = false;
  double alpha = 0.0;
  // THE DIAGNOSTIC, and it is NOT what gets blended any more. |T_gicp^-1 X(k)|
  // -- the ABSOLUTE difference between the smoother's estimate of scan k and
  // DLIO's own pose for scan k. Under the increment-1 design this WAS the
  // correction and it was a standing frame offset (49.2 mm at alpha = 0,
  // 0.33 mm/scan new). It stays on the record because it is the number the
  // sec 7 fixture reads: at alpha = 0 its p50 must be under 2 mm and its
  // lag-10 autocorrelation under 0.5, and a run where it is not has a gauge
  // problem whatever the trajectory says.
  double corr_m = 0.0,    corr_deg = 0.0;
  // WHAT THE NUDGE ACTUALLY AIMS AT: |T_gicp^-1 T_target|. Equal to corr under
  // (b) with the window re-anchored -- which is the point, the two agree only
  // when the frame offset is gone -- and equal to the one-scan increment
  // disagreement under (a).
  double target_m = 0.0,  target_deg = 0.0;
  double applied_m = 0.0, applied_deg = 0.0;   // what alpha actually moved
  int    kf_written = 0;
  int    kf_frozen_refused = 0;
  int    groups_written = 0;                   // MUST equal groups_expected
  // 6 normally, 5 when the bias write-back is off. The law compares the two
  // rather than a constant, so switching a group off cannot switch the law off.
  int    groups_expected = 6;
  bool   increment_mode = false;               // (a) was used, not (b)
  bool   bias_written = false;                 // group 5 actually ran
};

// b_baselink = R_bl_imu * b_sensor. R is proved against /ouster/metadata to
// 0.010 deg (imu_check/IMU_CHECK.md) and for rig 2860 it is diag(-1,-1,1), an
// involution -- which is exactly why the general form is written here and the
// involution is only ASSERTED in the fixture: a rig whose R is not an
// involution must not silently inherit this.
inline Eigen::Vector3f bias_bl_from_sensor(const Eigen::Matrix3f& R_bl_imu,
                                           const Eigen::Vector3d& b_sensor) {
  return R_bl_imu * b_sensor.cast<float>();
}
inline Eigen::Vector3d bias_sensor_from_bl(const Eigen::Matrix3f& R_bl_imu,
                                           const Eigen::Vector3f& b_bl) {
  return (R_bl_imu.transpose() * b_bl).cast<double>();
}

// THE SIX WRITES AT ONE LINE. Returns what it did; the caller enforces
// [[silent_no_op_law]] on the report, and test_smoother.cpp asserts each group
// individually with a sabotage that turns it red.
inline WriteBackReport write_back(const WriteTargets& t, const Solution& sol,
                                  double alpha, bool blend_increment = false) {
  WriteBackReport rep;
  rep.alpha = alpha;
  rep.increment_mode = blend_increment;
  if (!sol.valid) return rep;

  Eigen::Matrix4d T_gicp = Eigen::Matrix4d::Identity();
  if (t.T) T_gicp = t.T->cast<double>();
  pose_difference(T_gicp, sol.T, &rep.corr_m, &rep.corr_deg);

  // --- WHAT THE NUDGE AIMS AT (E4 increment 2) ------------------------------
  //
  // (b) THE DEFAULT -- sol.T itself, because the window it came out of is
  //     re-anchored on DLIO's own pose at its oldest surviving scan, so
  //     sol.T is DLIO's frame plus the window's own shape correction and
  //     nothing else. Nothing here changes; what changed is the graph.
  //
  // (a) THE ABLATION (dlio/smoother/blend_increment) -- compose the smoother's
  //     own increment onto DLIO's previous pose:
  //         T_target = T_dlio(k-1) . D_sm
  //                  = T_gicp(k) . D_dlio^-1 . D_sm
  //     which cancels every frame the two estimates might disagree about and
  //     leaves only the ONE SCAN of relative motion they disagree about. It is
  //     an ablation and not the default because it repairs the pose only: the
  //     keyframes, which the sixth write moves, are still absolute, so (a)
  //     also has to carry the same world->world gauge shift onto them (below)
  //     and the marginal it reports is still in the drifted frame.
  Eigen::Matrix4d T_target = sol.T;
#if DLIO_SMOOTHER_SABOTAGE == 9
  const bool gauge_shift = false;      // the map is left in the old frame
#else
  const bool gauge_shift = blend_increment && sol.rel_valid;
#endif
#if DLIO_SMOOTHER_SABOTAGE != 8
  if (blend_increment) {
    T_target = sol.rel_valid
                   ? Eigen::Matrix4d(T_gicp * (sol.T_rel_dlio.inverse() *
                                               sol.T_rel_smoother))
                   : T_gicp;     // scan 0: there is no increment yet
  }
#endif
  pose_difference(T_gicp, T_target, &rep.target_m, &rep.target_deg);
  // the world->world rigid shift (a) has to carry onto the map as well, so the
  // keyframes and the pose do not end up in two different frames. Under (b) it
  // is the identity by construction and is not computed at all, so (b)'s
  // keyframe arithmetic is bit-for-bit the increment-1 arithmetic.
  const Eigen::Matrix4d G = gauge_shift
                                ? Eigen::Matrix4d(T_target * sol.T.inverse())
                                : Eigen::Matrix4d::Identity();

  // alpha == 0: the smoother RUNS, LOGS EVERYTHING, AND WRITES NOTHING. This is
  // the mandatory A/A rung and the offline instrument in one, and it is the only
  // "byte identity" available under deterministic:false. Returning before any
  // write is what makes it byte-exact -- not a blend that happens to be the
  // identity, which would still round-trip the state through a chart.
#if DLIO_SMOOTHER_SABOTAGE != 4
  if (alpha == 0.0) return rep;
#endif

  const Eigen::Matrix4d T_hat = blend_pose(T_gicp, T_target, alpha);
  pose_difference(T_gicp, T_hat, &rep.applied_m, &rep.applied_deg);
  const Eigen::Matrix4f T_hat_f = T_hat.cast<float>();
  const Eigen::Matrix3f R_hat = T_hat_f.block<3, 3>(0, 0);
  const Eigen::Quaternionf q_hat = Eigen::Quaternionf(R_hat).normalized();
  const Eigen::Vector3f p_hat = T_hat_f.block<3, 1>(0, 3);

  // (1) T and T_corr. They ride together on purpose: publishCloud transforms
  //     the published cloud by T_corr, so correcting T alone would put the
  //     cloud somewhere the pose does not agree with.
  if (t.T) { *t.T = T_hat_f; ++rep.groups_written; }
#if DLIO_SMOOTHER_SABOTAGE != 2
  if (t.T_corr && t.T_prior) *t.T_corr = T_hat_f * t.T_prior->inverse();
#endif

  // (2) lidarPose -- THE ANCHOR. deskewPointcloud() builds the next scan's
  //     prior from lidarPose and geo.prev_vel, NOT from state.
#if DLIO_SMOOTHER_SABOTAGE != 1
  if (t.lidar_p && t.lidar_q) {
    *t.lidar_p = p_hat;
    *t.lidar_q = q_hat;
    ++rep.groups_written;
  }
#endif

  // (3) the observer's own state
  if (t.state_p && t.state_q) {
    *t.state_p = p_hat;
    *t.state_q = q_hat;
    ++rep.groups_written;
  }
  const Eigen::Vector3f v_prev = t.v_world ? *t.v_world : Eigen::Vector3f::Zero();
  const Eigen::Vector3f v_hat =
      (1.0f - (float)alpha) * v_prev + (float)alpha * sol.v.cast<float>();
  if (t.v_world) *t.v_world = v_hat;
  if (t.v_body)  *t.v_body  = R_hat.transpose() * v_hat;

  // (4) geo.prev_* -- the next prior's velocity
  if (t.geo_prev_vel) {
    *t.geo_prev_vel = v_hat;
    if (t.geo_prev_p) *t.geo_prev_p = p_hat;
    if (t.geo_prev_q) *t.geo_prev_q = q_hat;
    ++rep.groups_written;
  }

  // (5) the biases, converted out of the SENSOR frame the factor graph holds
  //     them in. No clamp is applied here: a random-walk prior at the measured
  //     Allan floor has no rail to pin against, which is the whole reason
  //     mechanism M-E stops being an amplifier.
#if DLIO_SMOOTHER_SABOTAGE == 11
  const bool do_bias = true;      // the flag is ignored: group 5 runs anyway
#else
  const bool do_bias = t.write_bias;
#endif
  rep.groups_expected = do_bias ? 6 : 5;
  if (do_bias && t.b_accel_bl && t.b_gyro_bl) {
    const Eigen::Vector3f ba = bias_bl_from_sensor(t.R_bl_imu, sol.b_accel);
    const Eigen::Vector3f bg = bias_bl_from_sensor(t.R_bl_imu, sol.b_gyro);
    *t.b_accel_bl = (1.0f - (float)alpha) * *t.b_accel_bl + (float)alpha * ba;
    *t.b_gyro_bl  = (1.0f - (float)alpha) * *t.b_gyro_bl  + (float)alpha * bg;
    ++rep.groups_written;
    rep.bias_written = true;
  }

  // (6) the in-window keyframes and the submap dirty flag. Without this the
  //     next align() measures against a map the smoother did not move and the
  //     correction is undone by the very measurement it was meant to reweigh.
#if DLIO_SMOOTHER_SABOTAGE != 3
  if (t.kf) {
    bool any = false;
    for (const auto& kp : sol.kf_poses) {
      Eigen::Matrix4d old;
      if (!t.kf->poseOf(kp.first, &old)) continue;
      const Eigen::Matrix4d kf_hat = gauge_shift ? Eigen::Matrix4d(G * kp.second)
                                                 : kp.second;
      const Eigen::Matrix4d target = blend_pose(old, kf_hat, alpha);
      double dm = 0.0, dd = 0.0;
      pose_difference(old, target, &dm, &dd);
      if (dm < t.kf_dirty_trans_m && dd < t.kf_dirty_rot_deg) continue;
      const Eigen::Matrix4d delta = target * old.inverse();
      if (!t.kf->queueDelta(kp.first, delta, target)) {
        ++rep.kf_frozen_refused;      // a frozen keyframe was offered a delta
        continue;
      }
      ++rep.kf_written;
      any = true;
    }
    if (any) t.kf->markSubmapDirty();
    ++rep.groups_written;   // the group ran, whether or not a keyframe qualified
  }
#endif

  rep.applied = true;
  return rep;
}

// ------------------------------------------------- silent-no-op counters ---

// COMPUTED = scans on which the smoother produced a correction above 1 mm.
// APPLIED  = scans on which the state actually moved.
// computed > 0 && applied == 0 is a HARD ERROR -- unless alpha == 0, where
// applied == 0 is the POINT and the law binds the other way round: the
// smoother must still have COMPUTED, or the control is measuring nothing.
struct NoOpLedger {
  long scans = 0;
  long solved = 0;
  long computed = 0;
  long applied = 0;
  long kf_written = 0;
  long kf_frozen_refused = 0;
  long exceptions = 0;
  long reseats = 0;
  // E4-REVIEW: scans on which the registration information was the CONSTANT
  // FLOOR -- no correspondences, or a Hessian the solve refused. On those scans
  // the graph holds an IMU chain and a 30 deg / 10 m prior and nothing else, so
  // the pose it produces IS the gyro's own answer. Counted, because the run
  // that made this counter necessary printed `verdict OK` on a 1,329 km
  // trajectory and its bend window was scored as a PASS.
  long floored = 0;
  // ... and the LONGEST unbroken run of them, against the number of poses the
  // window actually holds (window_vars / 3 -- MEASURED from the solve, never
  // assumed from lag_s and a scan rate). A streak longer than the window means
  // every variable in the window is now inertial: there is no scan left in it
  // that ever saw a correspondence.
  long floored_streak = 0;
  long floored_streak_max = 0;
  long window_poses = 0;
  // E4 INCREMENT 2, the law's THIRD clause: the smoother's ABSOLUTE marginal
  // against DLIO's own pose, counted only where the registration was actually
  // there. `measured` is the denominator (solved scans with the floor NOT
  // binding -- a window with no correspondences may legitimately disagree by
  // metres and that is not a gauge defect); `standing` is how many of those
  // exceeded the design's own 2 mm. Cheap: two counters, no history.
  long standing = 0;
  long measured = 0;
  double standing_max_m = 0.002;
  // and how many scans were re-anchored, so a run can show the clause ran
  long anchored = 0;
  // E4 INCREMENT 2, clause 4's counter: scans on which the RAW IMU TAP RAN DRY
  // -- not one sample inside the scan interval. The smoother then integrates a
  // single step at the nearest sample it has, which is declared behaviour for
  // one scan and a broken IMU chain for a hundred.
  long imu_dry = 0;
  // E4 INCREMENT 3: applied scans whose write ran a DIFFERENT number of groups
  // than the configuration expects -- five when six were due, or six when the
  // bias write-back is off. Both directions, because a flag that silently kept
  // writing would be the same defect as one that silently stopped.
  long groups_mismatch = 0;
  long groups_seen = 0;
  long groups_due = 0;

  void note(const Solution& sol, const WriteBackReport& rep) {
    ++scans;
    if (sol.anchor_added) ++anchored;
    if (sol.solved_this_scan && sol.imu_samples == 0) ++imu_dry;
    if (sol.solved_this_scan && sol.valid && !sol.floor_binding) {
      ++measured;
      if (rep.corr_m > standing_max_m) ++standing;
    }
    if (sol.solved_this_scan) ++solved;
    if (sol.solved_this_scan) {
      window_poses = sol.window_vars / 3;
      floored_streak = sol.floor_binding ? floored_streak + 1 : 0;
      floored_streak_max = std::max(floored_streak_max, floored_streak);
    }
    if (sol.solved_this_scan && sol.floor_binding) ++floored;
    if (sol.update_exception) ++exceptions;
    reseats = std::max(reseats, sol.reseats);
    if (sol.valid && rep.corr_m > 0.001) ++computed;
    if (rep.applied && (rep.applied_m > 0.0 || rep.applied_deg > 0.0)) ++applied;
    if (rep.applied) {
      groups_seen = rep.groups_written;
      groups_due = rep.groups_expected;
      if (rep.groups_written != rep.groups_expected) ++groups_mismatch;
    }
    kf_written += rep.kf_written;
    kf_frozen_refused += rep.kf_frozen_refused;
  }

  // "" when healthy; otherwise the message to put on stderr, once.
  std::string violation(double alpha, long min_scans = 500) const {
    if (scans < min_scans) return std::string();
    if (reseats > 0)
      return "the fixed-lag window was RE-SEATED " + std::to_string(reseats) +
             " time(s) after a failed update. The solve went indeterminant, the "
             "window's accumulated information was thrown away and the chain "
             "restarted. Look at dlio/smoother/linear_solver: on this bag "
             "MULTIFRONTAL_CHOLESKY squares a conditioning of ~4e7 into the "
             "normal equations and fails on healthy data, MULTIFRONTAL_QR does "
             "not. This run is not a clean arm.";
    if (solved == 0)
      return "the smoother was ENABLED for " + std::to_string(scans) +
             " scans and NEVER SOLVED: no window ever formed. Check that "
             "dlio/smoother/enabled and lag_s reached the node (rosparam dump "
             "/robot/dlio_odom), not just the recipe row.";
    // E4 INCREMENT 2, the law's FOURTH clause, and it is a BENCH the review will
    // never see that made it necessary. `raw_imu_keep_s` sizes the tap against
    // the WINDOW; what has to be covered is how far the SCAN pipeline is running
    // behind the IMU stream, and on a contended box that is unbounded. On an
    // increment-2 bench the solve reached 92 ms p50, DLIO fell more than 10 s
    // behind the player, and the tap ran DRY on 31 % of Sandbar's scans -- while
    // ncorr stayed at 20,000, so every registration column looked healthy. The
    // CombinedImuFactor was then one step at the nearest sample, the window's
    // velocity and attitude went wrong, the correction climbed 1.8 mm to 6 m in
    // eleven scans, the fixed-lag update threw, and the arm read 9.7 km on a
    // 27 m bag. Nothing in the trajectory said "the IMU tap ran dry"; the run
    // looked exactly like an estimator failure and was not one.
    // 1 % is not a fitted band: one dry scan is the declared fallback, a
    // hundred is a different sensor.  [[silent_no_op_law]]
#if DLIO_SMOOTHER_SABOTAGE != 10
    if (solved > 0 && imu_dry * 100 > solved)
      return "the RAW IMU TAP RAN DRY on " + std::to_string(imu_dry) + " of " +
             std::to_string(solved) +
             " solved scans -- not one sample inside the scan interval, so the "
             "CombinedImuFactor was a SINGLE STEP at the nearest sample the "
             "buffer still had. Every registration column will still look "
             "healthy while the window's velocity and attitude go wrong. The "
             "cause is the SCAN PIPELINE running further behind the IMU stream "
             "than dlio/smoother/raw_imu_keep_s covers -- a starved box, a "
             "contended bench, or a solve that no longer fits the scan period. "
             "Look at imu_n on the [SMOOTH] line and at solve_ms. This run is "
             "not a clean arm.";
#endif
    if (alpha == 0.0) {
      if (computed == 0)
        return "alpha = 0 and the smoother computed a correction on ZERO of " +
               std::to_string(scans) +
               " scans. The A/A control is measuring nothing: at alpha = 0 the "
               "smoother must still RUN and LOG, and only the write-back is "
               "skipped.";
      if (applied != 0)
        return "alpha = 0 and the state MOVED on " + std::to_string(applied) +
               " scans. The A/A control is not a control.";
      // E4 INCREMENT 2, the law's THIRD clause, and it is the one that would
      // have stopped increment 1 before a single node draw. At alpha = 0 the
      // smoother writes NOTHING, so its estimate of scan k cannot have been
      // moved by anything this code did: if it nevertheless sits far from
      // DLIO's own pose for scan k, on the MAJORITY of the scans where the
      // registration was actually present, then the quantity the write-back
      // blends at alpha > 0 is A STANDING FRAME OFFSET AND NOT A CORRECTION.
      // Increment 1 measured 49.2 mm of it with 0.33 mm/scan new, blended
      // alpha x that into lidarPose, state, geo.prev_* and the map ten times a
      // second, and ran 1,329 km before the OOM killer. The floor-binding
      // scans are excluded on purpose: a window with no correspondences in it
      // may legitimately disagree with the registration by metres, and that is
      // clause 2's subject, not this one.  [[silent_no_op_law]]
#if DLIO_SMOOTHER_SABOTAGE != 7
      if (measured > 0 && standing * 2 > measured)
        return "alpha = 0 and the smoother's ABSOLUTE marginal sat more than " +
               std::to_string(standing_max_m * 1e3) +
               " mm from DLIO's own pose on " + std::to_string(standing) +
               " of " + std::to_string(measured) +
               " scans that HAD registration -- the MAJORITY. At alpha = 0 the "
               "write-back writes nothing, so this cannot be something this "
               "code did: it is a STANDING FRAME OFFSET, and blending alpha x "
               "it into lidarPose, state, geo.prev_* and the map is injecting "
               "a constant bias every scan, not applying a correction. Check "
               "dlio/smoother/reanchor (the window must be pinned on DLIO's "
               "own pose at its oldest surviving scan) and "
               "dlio/smoother/anchor_sigma_pos_m.";
#endif
      return std::string();
    }
#if DLIO_SMOOTHER_SABOTAGE != 12
    if (groups_mismatch > 0)
      return "the write ran " + std::to_string(groups_seen) + " of the " +
             std::to_string(groups_due) +
             " write groups this configuration expects, on " +
             std::to_string(groups_mismatch) +
             " applied scan(s). Six are due normally and FIVE when "
             "dlio/smoother/writeback_bias is off -- the law compares the two, "
             "so switching a group off cannot switch the law off. A missing "
             "group means the correction reaches the state and not the thing "
             "the registration is measured AGAINST, and survives exactly one "
             "scan.";
#endif
    if (computed > 0 && applied == 0)
      return "the smoother computed a correction on " + std::to_string(computed) +
             " scans and the state NEVER MOVED (applied = 0). The nudge is a "
             "no-op: check alpha, and check that write_back() reached lidarPose "
             "and geo.prev_vel and not only state.";
    // E4-REVIEW, the law's SECOND clause. The first clause asks whether the
    // write HAPPENED. It does not ask whether there was anything to write it
    // from -- and that is not a hypothetical: on the E4 bench of big Sandland
    // the registration information was the constant floor on 3,132 of 5,947
    // solved scans (100 % of them inside the bend window that was then scored
    // as a PASS), the correspondence count reaching the graph was ZERO from
    // bag t 276, and the ledger still said verdict=OK on a trajectory 1,329 km
    // long. A majority is not a fitted band: when more than half the solved
    // scans carry no registration at all, the arm is inertial dead reckoning
    // with a smoother attached, and NO attitude or gravity number taken from it
    // is evidence about the arbitration -- the gyro arbiter shares its gyro and
    // its bias with the IMU factor, so such a run scores near zero BY
    // CONSTRUCTION.  [[silent_no_op_law]]
#if DLIO_SMOOTHER_SABOTAGE != 6
    // (i) the streak. This is the one that fires in TIME to matter: on the E4
    //     bench it trips at bag t 295.7 -- 44 s BEFORE the bend window opens --
    //     where the cumulative majority does not trip until t 565.6, long after
    //     the number it invalidates has been read. The comparison is against
    //     the window's OWN pose count, so there is no new constant here.
    if (window_poses > 0 && floored_streak_max > window_poses)
      return "the registration information was the CONSTANT FLOOR on " +
             std::to_string(floored_streak_max) +
             " CONSECUTIVE solved scans, against a window that holds " +
             std::to_string(window_poses) +
             " poses. Every variable in the window is inertial: not one scan "
             "left in it ever saw a correspondence. Any attitude, gravity or "
             "bend number taken from this run is the IMU's own answer read "
             "back by an arbiter that shares its gyro and its bias -- near "
             "zero BY CONSTRUCTION. Look at ncorr in the [DEGEN] record.";
    // (ii) and the run-level majority, for a run that loses the measurement in
    //      bursts rather than in one block.
    if (solved > 0 && floored * 2 > solved)
      return "the registration information was the CONSTANT FLOOR on " +
             std::to_string(floored) + " of " + std::to_string(solved) +
             " solved scans -- the MAJORITY. On those scans the graph held an "
             "IMU chain and a 30 deg / 10 m prior and no registration at all, "
             "so this arm is inertial dead reckoning and the gyro arbiter, "
             "which shares its gyro and its bias with that same IMU factor, "
             "scores it near zero BY CONSTRUCTION. No attitude, gravity or "
             "bend number from this run is evidence about the arbitration. "
             "Look at ncorr in the [DEGEN] record: a stock draw of this bag "
             "keeps 4,500-8,000 correspondences through the same windows.";
#endif
    return std::string();
  }
};

// ------------------------------------------------------------ the object ---

class Smoother {
 public:
  explicit Smoother(const Params& p);
  ~Smoother();

  // One scan in, one solution out. `kf_keys` are the keyframe indices whose
  // scan is THIS scan (updateKeyframes lays at most one per scan); they are
  // registered so a later solve can report their smoothed pose while they are
  // still inside the window.
  Solution update(const ScanInput& in);

  // E4 INCREMENT 2. THE POSE DLIO ACTUALLY ENDED UP HOLDING for the scan just
  // passed to update(), i.e. what write_back() wrote. Called from
  // smootherUpdate() immediately after the six writes, under the same geo.mtx.
  //
  // WHY IT IS NOT OPTIONAL. The re-anchor pins the window on "DLIO's own pose"
  // at the oldest surviving scan. update() only ever sees the pose BEFORE that
  // scan's write-back, so without this call the anchor would sit alpha x one
  // correction behind the chain DLIO is really running -- and the smoother
  // would re-derive, and re-apply, the same correction every scan for a whole
  // window. At alpha = 0 write_back() writes nothing and T_applied is
  // identically T_gicp, so the A/A control is untouched either way.
  void noteApplied(const Eigen::Matrix4d& T_applied);

  // Register a keyframe laid at the scan just passed to update(). Called from
  // updateKeyframes(), which runs after getNextPose().
  void noteKeyframe(int kf_index);

  // The bias prior, in the SENSOR frame. Seeded from the run's own 3 s init
  // calibration (DLIO's state.b after callbackImu's calibration branch),
  // rotated out of baselink. Must be called before the first update().
  void setBiasPrior(const Eigen::Vector3d& accel_sensor,
                    const Eigen::Vector3d& gyro_sensor);

  // baselink <- imu, from the GENERATED yaml (never the bundle sidecar, which
  // still ships the house7 MicroStrain lever arm).
  void setExtrinsic(const Eigen::Matrix3d& R_bl_imu, const Eigen::Vector3d& t_bl_imu);

  // OFFLINE ONLY, and the node never calls it. After each update, every scan
  // key still inside the window has a smoothed estimate; a caller that
  // overwrites its own array with these on every scan ends up holding, for each
  // scan, the LAST estimate before that scan left the window -- which is what
  // the offline refits (e0_solve / e2_solve) publish as "the smoothed
  // trajectory". The live nudge cannot use it: at scan k the only estimate that
  // exists for scan k is the lag-0 one, and that is what write_back() applies.
  // Both series are written by the replay so the A/A compares like with like.
  int readWindow(std::vector<long>* keys,
                 std::vector<Eigen::Matrix4d>* poses,
                 std::vector<Eigen::Vector3d>* vels) const;

  const Params& params() const { return params_; }
  long scans() const;
  bool gtsamLinked() const;      // asserted at build time, reported at run time
  std::string versionString() const;

 private:
  Params params_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace smoother
}  // namespace dlio

#endif  // DLIO_SMOOTHER_H
