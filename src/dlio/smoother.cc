// E4 / ARCHITECT A -- the fixed-lag smoother itself. See include/dlio/smoother.h
// for the architecture, the frames and the six writes; this file is the graph.
//
// THE GRAPH IS E0'S GRAPH AND E2'S INFORMATION, and that is deliberate: the
// offline A/A (E4/IMPL/impl_aa.py) replays a recorded draw's own measurements
// through THIS code and requires it to reproduce the offline refit. A mismatch
// there is a bug, not a result -- so every constant, every noise model and the
// solver itself are the same ones e0_solve.py / e2_solve.py use, and where they
// differ it is written down in IMPL.md rather than discovered later.
//
//   X(k) : Pose3      world <- baselink, one per SCAN (not per keyframe)
//   V(k) : Vector3    world-frame linear velocity
//   B(k) : imuBias    IMU SENSOR frame
//
//   F1  BetweenFactor<Pose3>(X(k-1), X(k), dp6_k,
//                            Robust(Huber(1.345), Gaussian::Information(Lambda)))
//       Lambda = s * H6b + FLOOR. HUBER, CONVEX, NEVER REDESCENDING: a cold
//       Geman-McClure weighted a hand-placed 12.52 m closure at 2.46e-07 and
//       "can never switch it back on" (GLIM_LC.md).
//   F2  CombinedImuFactor over the RAW samples in (t_{k-1}, t_k], at the rig's
//       own measured Allan noise, with NO BIAS CLAMP OF ANY KIND. The bias
//       random walk lives inside the same factor.
//   F3  the gauge prior on X(0)/V(0)/B(0) -- the gravity/attitude anchoring:
//       tight in roll and pitch, loose but non-singular in yaw and x/y/z.
//   F3' THE RE-ANCHOR (E4 increment 2). A TIGHT prior on the oldest pose that
//       will SURVIVE this update, at DLIO's own value for that scan. Without
//       it F3's 10 m gauge lets the window's absolute placement drift away
//       from DLIO's chain and stay there -- measured at alpha = 0, where the
//       write-back writes nothing: 49.2 mm standing, 0.33 mm/scan new, lag-10
//       autocorrelation 0.988 (E4/REVIEW.md sec 8.1) -- and the write-back
//       then blends THAT into the filter every scan. With it, the window's
//       gauge IS DLIO's frame and `corr` measures only the window's own
//       disagreement about the shape of the last lag_s seconds.
//   F4  the marginalisation prior, produced by the fixed-lag smoother itself.
//       Its spectrum goes on the [SMOOTH] line from day one, because
//       new-ct.md R3 names it as "the fourth sink" in a programme that has
//       already produced three.
//
// SOLVER: BatchFixedLagSmoother (Levenberg-Marquardt). NOT ISAM2 -- E0's
// deviation 2, measured: ISAM2 is undamped Gauss-Newton and inside the freeze
// (bag t 353-382), where the input chain and the IMU disagree by metres, it
// overshoots to 2.8e4 m/s by t 373 and goes indeterminant at t 384.9. QR did
// not help; it is divergence, not conditioning.

#include "dlio/smoother.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>

namespace {

inline gtsam::Key KX(long k) { return gtsam::Symbol('x', (std::uint64_t)k); }
inline gtsam::Key KV(long k) { return gtsam::Symbol('v', (std::uint64_t)k); }
inline gtsam::Key KB(long k) { return gtsam::Symbol('b', (std::uint64_t)k); }

inline gtsam::Pose3 pose_of(const Eigen::Matrix4d& T) {
  return gtsam::Pose3(gtsam::Rot3(T.block<3, 3>(0, 0)), T.block<3, 1>(0, 3));
}
inline Eigen::Matrix4d mat_of(const gtsam::Pose3& p) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = p.rotation().matrix();
  T.block<3, 1>(0, 3) = p.translation();
  return T;
}

// The sigmas of the prior that seeds a chain -- at k = 0 and after a re-seat.
// With the re-anchor ON this IS the anchor: X(0) is pinned at DLIO's own pose,
// so the window's gauge is DLIO's frame from the first scan and never has a
// 10 m free placement to drift into. With it OFF this is E0's own gauge prior,
// value for value, which is what the increment-1 arm ran.
inline Eigen::Matrix<double, 6, 1> gauge_sigmas(const dlio::smoother::Params& P) {
  Eigen::Matrix<double, 6, 1> gs;
  if (P.reanchor && P.anchor_in_graph) {
    const double r = P.anchor_sigma_rot_deg * M_PI / 180.0;
    gs << r, r, r, P.anchor_sigma_pos_m, P.anchor_sigma_pos_m,
        P.anchor_sigma_pos_m;
  } else {
    gs << P.gauge_sigma_rp_deg * M_PI / 180.0,
        P.gauge_sigma_rp_deg * M_PI / 180.0,
        P.gauge_sigma_yaw_deg * M_PI / 180.0, P.gauge_sigma_pos_m,
        P.gauge_sigma_pos_m, P.gauge_sigma_pos_m;
  }
  return gs;
}

}  // namespace

namespace dlio {
namespace smoother {

struct Smoother::Impl {
  explicit Impl(const Params& p) : par(p) {
    // --- the preintegration parameters, E0's imu_params() line for line -----
    auto pp = gtsam::PreintegrationCombinedParams::MakeSharedU(p.gravity);
    Eigen::Matrix3d ac = Eigen::Matrix3d::Zero(), gc = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d bac = Eigen::Matrix3d::Zero(), bgc = Eigen::Matrix3d::Zero();
    const double kappa = std::sqrt(2.0 / 3.0);   // ADEV(tau=2s) -> sigma_rw
    for (int i = 0; i < 3; ++i) {
      ac(i, i) = p.vrw[i] * p.vrw[i];
      gc(i, i) = p.arw[i] * p.arw[i];
      const double ba = p.acc_adev_2s[i] / kappa;
      const double bg = p.gyr_adev_2s[i] / kappa;
      bac(i, i) = ba * ba;
      bgc(i, i) = bg * bg;
    }
    pp->setAccelerometerCovariance(ac);
    pp->setGyroscopeCovariance(gc);
    pp->setBiasAccCovariance(bac);
    pp->setBiasOmegaCovariance(bgc);
    pp->setIntegrationCovariance(Eigen::Matrix3d::Identity() *
                                 (p.integration_sigma * p.integration_sigma));
    Eigen::Matrix<double, 6, 6> bi = Eigen::Matrix<double, 6, 6>::Zero();
    bi.diagonal() << p.bias_init_acc * p.bias_init_acc,
        p.bias_init_acc * p.bias_init_acc, p.bias_init_acc * p.bias_init_acc,
        p.bias_init_gyr * p.bias_init_gyr, p.bias_init_gyr * p.bias_init_gyr,
        p.bias_init_gyr * p.bias_init_gyr;
    pp->setBiasAccOmegaInit(bi);
    pim_params = pp;
    setExtrinsic(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    lm.setMaxIterations(p.lm_max_iterations);
    lm.setRelativeErrorTol(p.lm_relative_error_tol);
    lm.setLinearSolverType(p.linear_solver);
    sm.reset(new gtsam::BatchFixedLagSmoother(p.lag_s, lm));
  }

  void setExtrinsic(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    pim_params->setBodyPSensor(gtsam::Pose3(gtsam::Rot3(R), t));
  }

  Params par;
  gtsam::LevenbergMarquardtParams lm;
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> pim_params;
  std::unique_ptr<gtsam::BatchFixedLagSmoother> sm;

  gtsam::imuBias::ConstantBias bias0;
  long k = -1;
  double t0 = 0.0;
  long scans = 0;

  // accumulated across marginalize_every scans
  gtsam::NonlinearFactorGraph pending_f;
  gtsam::Values pending_v;
  gtsam::FixedLagSmoother::KeyTimestampMap pending_t;

  // the last factor of each kind that was ADDED, so its residual can be read
  gtsam::NonlinearFactor::shared_ptr last_imu, last_reg;

  // keyframe index -> the scan key its pose IS
  std::unordered_map<int, long> kf_key;
  long kf_frozen_upto = -1;      // keyframes with key <= this have left the window
  bool exc_reported = false;
  // A fixed-lag update that throws leaves the window in a state nobody can
  // describe: the keys of that scan may or may not be in it, and every later
  // CombinedImuFactor references the previous scan's keys. Left alone, ONE
  // exception turns into a cascade -- the window stops marginalising, grows
  // without bound and the solve time climbs, and NOTHING in the trajectory
  // says so (measured: 309 throws in 2,600 scans took the window from 51 scans
  // to 861). So a throw ARMS A RE-SEAT: the next scan starts a fresh chain,
  // loudly, with a counter on the [SMOOTH] line.
  bool needs_reseat = false;
  long reseats = 0;
  long singular_info = 0;
  bool singular_reported = false;

  // --- E4 INCREMENT 2: the re-anchor ---------------------------------------
  // DLIO's OWN pose for every scan key still live, and that key's window
  // timestamp. Both are pruned to the window on every update, so this is
  // O(window), not O(run).
  std::map<long, Eigen::Matrix4d> dlio_pose;
  std::map<long, double> key_time;
  long anchored_upto = -1;   // the newest key that has already been anchored
};

Smoother::Smoother(const Params& p) : params_(p), impl_(new Impl(p)) {}
Smoother::~Smoother() = default;

void Smoother::setBiasPrior(const Eigen::Vector3d& a, const Eigen::Vector3d& g) {
  impl_->bias0 = gtsam::imuBias::ConstantBias(a, g);
}

void Smoother::setExtrinsic(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  impl_->setExtrinsic(R, t);
}

void Smoother::noteApplied(const Eigen::Matrix4d& T_applied) {
  if (impl_->k < 0) return;
  if (!T_applied.allFinite()) return;      // a broken write never becomes an anchor
  impl_->dlio_pose[impl_->k] = T_applied;
}

void Smoother::noteKeyframe(int kf_index) {
  if (impl_->k < 0) return;
  impl_->kf_key[kf_index] = impl_->k;
}

int Smoother::readWindow(std::vector<long>* keys,
                         std::vector<Eigen::Matrix4d>* poses,
                         std::vector<Eigen::Vector3d>* vels) const {
  if (keys) keys->clear();
  if (poses) poses->clear();
  if (vels) vels->clear();
  int n = 0;
  const auto& stamps = impl_->sm->timestamps();
  for (long k = 0; k <= impl_->k; ++k) {
    if (stamps.find(KX(k)) == stamps.end()) continue;
    try {
      const gtsam::Pose3 X = impl_->sm->calculateEstimate<gtsam::Pose3>(KX(k));
      const gtsam::Vector3 V = impl_->sm->calculateEstimate<gtsam::Vector3>(KV(k));
      if (keys) keys->push_back(k);
      if (poses) poses->push_back(mat_of(X));
      if (vels) vels->push_back(V);
      ++n;
    } catch (const std::exception& e) {
      // a key the smoother says is in the window but cannot produce is a
      // defect; the caller sees a short list rather than a wrong pose
    }
  }
  return n;
}

long Smoother::scans() const { return impl_->scans; }
bool Smoother::gtsamLinked() const { return true; }
std::string Smoother::versionString() const {
  return std::string("gtsam ") + std::to_string(GTSAM_VERSION_MAJOR) + "." +
         std::to_string(GTSAM_VERSION_MINOR) + "." +
         std::to_string(GTSAM_VERSION_PATCH) + " BatchFixedLagSmoother(LM)";
}

Solution Smoother::update(const ScanInput& in) {
  Solution out;
  Impl& I = *impl_;
  const Params& P = params_;
  const auto t_begin = std::chrono::steady_clock::now();
  ++I.scans;

  // ---- F2: the preintegration over the RAW samples in (prev, stamp] --------
  // Mirrors e0_solve.build_pims() exactly, including the trailing step that
  // carries the last sample forward to the scan stamp. A sample interval longer
  // than imu_gap_s is COUNTED and still integrated (declared, not hidden): DLIO
  // itself has no gap handling at all and the record must be able to say the
  // gap fired zero times rather than not existing.
  gtsam::PreintegratedCombinedMeasurements pim(I.pim_params, I.bias0);
  {
    double prev = in.prev_stamp;
    for (const auto& s : in.imu) {
      if (s.stamp <= in.prev_stamp || s.stamp > in.stamp) continue;
      double dt = s.stamp - prev;
      prev = s.stamp;
      if (dt <= 0.0) continue;
      if (dt > P.imu_gap_s) ++out.imu_gaps;
      pim.integrateMeasurement(s.accel, s.gyro, dt);
      ++out.imu_samples;
    }
    if (out.imu_samples == 0) {
      // No sample landed inside the interval. Integrate ONE step at the nearest
      // sample the buffer has rather than leave the chain broken -- and never
      // at zero, which would read as free fall. Counted as a gap so the record
      // can say it happened. (On the fixed player this is moot: 640.00 Hz,
      // 0 gaps, p50 64 samples per scan, min 57.)
      const double dt = std::max(in.stamp - in.prev_stamp, 1e-4);
      ImuSample s;
      if (!in.imu.empty()) {
        s = in.imu.front();
        double best = std::abs(s.stamp - in.stamp);
        for (const auto& c : in.imu) {
          const double d = std::abs(c.stamp - in.stamp);
          if (d < best) { best = d; s = c; }
        }
      }
      pim.integrateMeasurement(s.accel, s.gyro, dt);
      ++out.imu_gaps;
    } else {
      const double dt = in.stamp - prev;
      if (dt > 1e-9) {
        const auto& s = in.imu.back();
        pim.integrateMeasurement(s.accel, s.gyro, dt);
      }
    }
  }

  if (!in.imu.empty()) out.imu_lead_s = in.imu.back().stamp - in.stamp;

  const gtsam::Pose3 Tg = pose_of(in.T_gicp);
  Eigen::Vector3d v_init = in.v_world;
  if (!v_init.allFinite()) v_init.setZero();

  if (I.k < 0) {
    // ---- F3: the gauge prior. TIGHT in roll/pitch (gravity is the witness),
    // loose but non-singular in yaw and x/y/z. This IS the attitude anchoring:
    // the world frame is gravity-aligned by DLIO's own init and the
    // accelerometer constrains roll and pitch inside every F2 factor, so a
    // second gravity opinion inside the window would be an unforced risk
    // (A_nudge sec 3.3 F5, deliberately not built).
    const Eigen::Matrix<double, 6, 1> gs = gauge_sigmas(P);
    Eigen::Matrix<double, 6, 1> bs;
    bs << P.bias_prior_sigma_accel, P.bias_prior_sigma_accel,
        P.bias_prior_sigma_accel, P.bias_prior_sigma_gyro,
        P.bias_prior_sigma_gyro, P.bias_prior_sigma_gyro;
    I.k = 0;
    I.t0 = in.stamp;
    if (P.reanchor && P.anchor_in_graph) I.anchored_upto = 0;   // X(0) IS the anchor
    I.pending_f.addPrior(KX(0), Tg, gtsam::noiseModel::Diagonal::Sigmas(gs));
    I.pending_f.addPrior(KV(0), gtsam::Vector3(v_init),
                         gtsam::noiseModel::Isotropic::Sigma(3, P.v0_sigma));
    I.pending_f.addPrior(KB(0), I.bias0, gtsam::noiseModel::Diagonal::Sigmas(bs));
    I.pending_v.insert(KX(0), Tg);
    I.pending_v.insert(KV(0), gtsam::Vector3(v_init));
    I.pending_v.insert(KB(0), I.bias0);
    I.pending_t[KX(0)] = 0.0;
    I.pending_t[KV(0)] = 0.0;
    I.pending_t[KB(0)] = 0.0;
    I.key_time[0] = 0.0;
    I.dlio_pose[0] = in.T_gicp;
  } else if (I.needs_reseat) {
    // RE-SEAT. Bounded and reported, never silent. The key numbering continues
    // so the keyframe map and the log stay readable; what restarts is the
    // graph. Every keyframe the old window owned is frozen by construction --
    // its key no longer exists anywhere -- and is reported as frozen so the map
    // side can refuse any later delta for it.
    const long k = ++I.k;
    I.sm.reset(new gtsam::BatchFixedLagSmoother(P.lag_s, I.lm));
    I.needs_reseat = false;
    ++I.reseats;
    for (const auto& kv : I.kf_key) out.kf_frozen.push_back(kv.first);
    I.kf_key.clear();
    I.key_time.clear();
    if (P.reanchor && P.anchor_in_graph) I.anchored_upto = k;   // X(k) IS the anchor
    const Eigen::Matrix<double, 6, 1> gs = gauge_sigmas(P);
    Eigen::Matrix<double, 6, 1> bs;
    bs << P.bias_prior_sigma_accel, P.bias_prior_sigma_accel,
        P.bias_prior_sigma_accel, P.bias_prior_sigma_gyro,
        P.bias_prior_sigma_gyro, P.bias_prior_sigma_gyro;
    I.pending_f.addPrior(KX(k), Tg, gtsam::noiseModel::Diagonal::Sigmas(gs));
    I.pending_f.addPrior(KV(k), gtsam::Vector3(v_init),
                         gtsam::noiseModel::Isotropic::Sigma(3, P.v0_sigma));
    I.pending_f.addPrior(KB(k), I.bias0, gtsam::noiseModel::Diagonal::Sigmas(bs));
    I.pending_v.insert(KX(k), Tg);
    I.pending_v.insert(KV(k), gtsam::Vector3(v_init));
    I.pending_v.insert(KB(k), I.bias0);
    const double tk = in.stamp - I.t0;
    I.pending_t[KX(k)] = tk;
    I.pending_t[KV(k)] = tk;
    I.pending_t[KB(k)] = tk;
    I.key_time[k] = tk;
    I.dlio_pose[k] = in.T_gicp;
    I.last_imu.reset();
    I.last_reg.reset();
    std::fprintf(stderr,
                 "[SMOOTH][ERROR] RE-SEATING the fixed-lag window at scan k=%ld "
                 "(t=%.4f) after a failed update -- re-seat #%ld. The window's "
                 "accumulated information is GONE from here and the chain starts "
                 "again at this scan's registration pose with the gauge prior. "
                 "This is bounded and counted; a run with re-seats in it is not "
                 "a clean arm.\n",
                 (long)k, in.stamp, (long)I.reseats);
    std::fflush(stderr);
  } else {
    const long k = ++I.k;
    I.last_imu.reset(new gtsam::CombinedImuFactor(KX(k - 1), KV(k - 1), KX(k),
                                                  KV(k), KB(k - 1), KB(k), pim));
    I.pending_f.add(I.last_imu);

    // ---- F1: the registration factor ------------------------------------
    I.last_reg.reset();
    if (in.dp6_valid) {
      bool proj = false, bind = false;
      const Eigen::Matrix<double, 6, 6> Lam = information_from_h6b(
          in.H6b, in.h_valid, P.info_scale_rot, P.info_scale_trans,
          P.floor_sigma_rot_deg, P.floor_sigma_trans_m, &proj, &bind);
      out.psd_projected = proj;
      out.floor_binding = bind;
      // A SINGULAR information is not a weak factor, it is a broken one: it
      // makes the linear system rank-deficient and the trajectory becomes pure
      // inertial dead reckoning while every other column still looks healthy.
      // The constant floor exists so this cannot happen; if it happens anyway
      // the factor is REFUSED and COUNTED, never added.
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> esL(Lam);
      if (esL.info() != Eigen::Success || esL.eigenvalues()(0) <= 0.0) {
        ++I.singular_info;
        if (!I.singular_reported) {
          I.singular_reported = true;
          std::fprintf(stderr,
                       "[SMOOTH][ERROR] the registration information is "
                       "SINGULAR at scan k=%ld (lambda_min=%.6g). The factor is "
                       "REFUSED, not added with zero weight. Check "
                       "dlio/smoother/floor_sigma_* and info_scale_*: a floor "
                       "of zero with an invalid Hessian is exactly this.\n",
                       (long)k, esL.info() == Eigen::Success
                                    ? esL.eigenvalues()(0) : 0.0);
          std::fflush(stderr);
        }
        goto no_reg_factor;
      }
      {
      const gtsam::Pose3 meas(gtsam::Rot3(exp_so3(in.dp6.head<3>())),
                              gtsam::Point3(in.dp6.tail<3>()));
      auto base = gtsam::noiseModel::Gaussian::Information(Lam);
      gtsam::SharedNoiseModel model = base;
      // huber_k <= 0 removes the robust kernel entirely. It is NOT a shipping
      // configuration -- the node refuses it -- and it exists so the offline
      // harness can attribute a difference to the kernel rather than guess.
      if (P.huber_k > 0.0) {
        model = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(P.huber_k), base);
      }
      I.last_reg.reset(
          new gtsam::BetweenFactor<gtsam::Pose3>(KX(k - 1), KX(k), meas, model));
      I.pending_f.add(I.last_reg);
      }
      no_reg_factor:;
    }

    I.pending_v.insert(KX(k), Tg);
    I.pending_v.insert(KV(k), gtsam::Vector3(v_init));
    I.pending_v.insert(KB(k), I.bias0);
    const double tk = in.stamp - I.t0;
    I.pending_t[KX(k)] = tk;
    I.pending_t[KV(k)] = tk;
    I.pending_t[KB(k)] = tk;
    I.key_time[k] = tk;
    I.dlio_pose[k] = in.T_gicp;
  }

  const long k = I.k;
  const int every = std::max(1, P.marginalize_every);
  const bool do_solve = (k == 0) || ((k % every) == 0);

  // ---- F3': RE-ANCHOR THE WINDOW ON DLIO'S OWN POSE (E4 increment 2) -------
  //
  // The oldest pose that will SURVIVE this update gets a TIGHT prior at DLIO's
  // own value for that scan. gtsam's BatchFixedLagSmoother marginalises every
  // key whose timestamp is strictly older than (newest - lag), so the survivor
  // is the smallest key at or after that cut. Each key is anchored AT MOST ONCE
  // -- the cut advances monotonically, so exactly one anchor is live at a time
  // and the previous one has already been absorbed into the marginalisation
  // prior. Adding it repeatedly to the same key would multiply its information
  // by the number of scans it stayed oldest, which is not a prior, it is a
  // count.
  //
  // WHAT IT DOES NOT DO: it does not pin the window's SHAPE. Every pose newer
  // than the anchor is as free as it was, so the arbitration -- the IMU chain
  // against the registration, over lag_s seconds -- is unchanged. What it
  // removes is the one degree of freedom the design never wanted: where the
  // whole window sits.
  if (do_solve && P.reanchor && P.anchor_in_graph && k > 0) {
    const double tk = in.stamp - I.t0;
    const double t_cut = tk - P.lag_s;
    long k_anchor = -1;
    for (const auto& kv : I.key_time) {
      if (kv.second >= t_cut) { k_anchor = kv.first; break; }
    }
    // k_anchor == k would pin THIS scan, i.e. hand the whole window back to
    // the registration; that can only happen if the lag is shorter than one
    // scan period, and it is refused rather than silently applied.
    if (k_anchor >= 0 && k_anchor < k && k_anchor > I.anchored_upto &&
        I.dlio_pose.count(k_anchor)) {
      const gtsam::Pose3 Xa = pose_of(I.dlio_pose[k_anchor]);
      Eigen::Matrix<double, 6, 1> as;
      const double ar = P.anchor_sigma_rot_deg * M_PI / 180.0;
      as << ar, ar, ar, P.anchor_sigma_pos_m, P.anchor_sigma_pos_m,
          P.anchor_sigma_pos_m;
      I.pending_f.addPrior(KX(k_anchor), Xa,
                           gtsam::noiseModel::Diagonal::Sigmas(as));
      I.anchored_upto = k_anchor;
      out.anchor_key = k_anchor;
      out.anchor_added = true;
      // how far the anchor had to move the window, BEFORE the solve -- the
      // gauge drift this clause is removing, per scan, on the record.
      try {
        const Eigen::Matrix4d Xw =
            mat_of(I.sm->calculateEstimate<gtsam::Pose3>(KX(k_anchor)));
        pose_difference(I.dlio_pose[k_anchor], Xw, &out.anchor_resid_m,
                        &out.anchor_resid_deg);
      } catch (const std::exception& e) {
        out.anchor_resid_m = out.anchor_resid_deg = -1.0;
      }
    }
  }

  if (do_solve) {
    try {
      const gtsam::FixedLagSmoother::Result r =
          I.sm->update(I.pending_f, I.pending_v, I.pending_t);
      out.lm_iterations = (int)r.iterations;
      out.window_vars = (int)r.nonlinearVariables + (int)r.linearVariables;
    } catch (const std::exception& e) {
      out.update_exception = true;
      I.needs_reseat = true;
      // The FIRST one, NAMED. An estimator that throws once and then quietly
      // stops marginalising looks like a working run with a growing window and
      // a growing solve time, and nothing in the trajectory says so.
      if (!I.exc_reported) {
        I.exc_reported = true;
        std::fprintf(stderr,
                     "[SMOOTH][ERROR] the fixed-lag update THREW at scan k=%ld "
                     "(t=%.4f): %s\n",
                     (long)I.k, in.stamp, e.what());
        std::fflush(stderr);
      }
    }
    I.pending_f.resize(0);
    I.pending_v.clear();
    I.pending_t.clear();
    out.solved_this_scan = true;
  }

  // ---- read the marginal for THIS scan ------------------------------------
  // T_graph is the answer IN THE WINDOW'S OWN FRAME and out.T is the answer in
  // DLIO's. They differ by the gauge below, and every quantity that has to be
  // consistent with the GRAPH -- the two factor residuals, the window's own
  // increment -- is computed from T_graph, never from out.T.
  Eigen::Matrix4d T_graph = Eigen::Matrix4d::Identity();
  try {
    const gtsam::Pose3 Xk = I.sm->calculateEstimate<gtsam::Pose3>(KX(k));
    const gtsam::Vector3 Vk = I.sm->calculateEstimate<gtsam::Vector3>(KV(k));
    const gtsam::imuBias::ConstantBias Bk =
        I.sm->calculateEstimate<gtsam::imuBias::ConstantBias>(KB(k));
    out.T = mat_of(Xk);
    T_graph = out.T;
    out.v = Vk;
    out.b_accel = Bk.accelerometer();
    out.b_gyro = Bk.gyroscope();
    out.valid = out.T.allFinite() && out.v.allFinite();
  } catch (const std::exception& e) {
    out.valid = false;
  }

  // ---- F3'' RE-ANCHOR AS A GAUGE TRANSFORM (the default) -------------------
  //
  // The window's own gauge is left exactly as E0 built it -- loose, and free to
  // wander, which inside a fixed-lag window is the correct thing for it to be.
  // What is fixed is the ANSWER: the window is re-expressed in DLIO's own frame
  // by the rigid transform that carries the window's estimate of its oldest
  // surviving pose onto DLIO's own pose for that same scan.
  //
  //     G   = T_dlio(k_a) . X(k_a)^-1                (world <- world)
  //     X_out(k) = G . X(k),   v_out = R_G . v,   kf_out = G . kf
  //
  // At k = k_a this is an identity by construction, which is the sense in which
  // the standing offset is zero: the window and DLIO agree EXACTLY about where
  // the window starts, and `corr` at the newest scan is then the window's own
  // disagreement about the SHAPE of the last lag_s seconds and nothing else.
  //
  // Pose, velocity and keyframes are transformed TOGETHER -- a gauge that moved
  // the pose and left the map behind would be the same defect in a new costume.
  // marg_cov is a variance in the local tangent and G is a rigid motion, so the
  // translation block is rotated, not scaled; it is left as the graph reported
  // it and the record says so in its units token.
  if (out.valid && P.reanchor && !P.anchor_in_graph && k > 0) {
    const double tk = in.stamp - I.t0;
    const double t_cut = tk - P.lag_s;
    long k_a = -1;
    for (const auto& kv : I.key_time) {
      if (kv.second >= t_cut && I.dlio_pose.count(kv.first)) { k_a = kv.first; break; }
    }
    if (k_a >= 0 && k_a < k) {
      try {
        const Eigen::Matrix4d Xa =
            mat_of(I.sm->calculateEstimate<gtsam::Pose3>(KX(k_a)));
        const Eigen::Matrix4d G = I.dlio_pose[k_a] * Xa.inverse();
        if (G.allFinite()) {
          pose_difference(I.dlio_pose[k_a], Xa, &out.anchor_resid_m,
                          &out.anchor_resid_deg);
          out.T = G * out.T;
          out.v = G.block<3, 3>(0, 0) * out.v;
          out.anchor_key = k_a;
          out.anchor_added = true;
          out.anchor_gauge = G;
          out.valid = out.T.allFinite() && out.v.allFinite();
        }
      } catch (const std::exception& e) {
        // the oldest survivor cannot be read: leave the answer in the window's
        // own frame rather than in a frame nobody can name, and say so
        out.anchor_key = -2;
      }
    }
  }

  // ---- the window's own INCREMENT over the last scan, and DLIO's ----------
  // Both gauge-free: whatever frame the window is in cancels in X(k-1)^-1 X(k),
  // and DLIO's own frame cancels in T_{k-1}^-1 T_k. This is what option (a)
  // blends, and it is on the record either way so a reader can see how much of
  // `corr` was the frame and how much was the motion. It costs one estimate
  // read of a key that is already in the window.
  if (out.valid && k > 0 && I.dlio_pose.count(k - 1) && I.dlio_pose.count(k)) {
    try {
      const Eigen::Matrix4d Xkm =
          mat_of(I.sm->calculateEstimate<gtsam::Pose3>(KX(k - 1)));
      const Eigen::Matrix4d& Dp = I.dlio_pose[k - 1];
      out.T_rel_smoother = Xkm.inverse() * T_graph;
      out.T_rel_dlio = Dp.inverse() * I.dlio_pose[k];
      out.rel_valid = out.T_rel_smoother.allFinite() && out.T_rel_dlio.allFinite();
    } catch (const std::exception& e) {
      out.rel_valid = false;      // the previous key has left the window
    }
  }

  // ---- the marginal's own spectrum. new-ct.md R3 names the marginalisation
  // prior as "the fourth sink" in a programme that has already produced three,
  // and its tell is the marginal going overconfident along a direction the
  // window never observed. It goes on the [SMOOTH] line from day one.
  //
  // BatchFixedLagSmoother::marginalCovariance() is DECLARED and THROWS
  // "not implemented" in gtsam 4.2.2, which is the kind of thing that becomes a
  // column of -1 nobody reads. So the marginal is taken the honest way, from
  // the smoother's OWN window graph and its OWN estimate -- which includes the
  // marginalisation prior, i.e. exactly the quantity that has to be watched.
  // Null factors (the ones the lag has removed) are dropped first: Marginals
  // will not eliminate a graph that contains them.
  if (out.valid && (k % std::max(1, P.marg_every)) == 0) {
    try {
      const gtsam::NonlinearFactorGraph& g = I.sm->getFactors();
      gtsam::NonlinearFactorGraph live;
      live.reserve(g.size());
      for (const auto& f : g) if (f) live.push_back(f);
      gtsam::Marginals marg(live, I.sm->calculateEstimate(),
                            gtsam::Marginals::CHOLESKY);
      const gtsam::Matrix C = marg.marginalCovariance(KX(k));
      if (C.rows() == 6 && C.cols() == 6)
        for (int i = 0; i < 6; ++i) out.marg_cov[i] = C(i, i);
    } catch (const std::exception& e) {
      for (int i = 0; i < 6; ++i) out.marg_cov[i] = -1.0;   // said, not hidden
    }
  } else if (out.valid) {
    for (int i = 0; i < 6; ++i) out.marg_cov[i] = 0.0;      // not asked for
  }

  // ---- the two residuals, so the run can show WHICH factor moved -----------
  if (out.valid && (I.last_imu || I.last_reg)) {
    try {
      gtsam::Values v;
      const long kp = std::max(0L, k - 1);
      v.insert(KX(k), pose_of(T_graph));
      v.insert(KV(k), gtsam::Vector3(
                          impl_->sm->calculateEstimate<gtsam::Vector3>(KV(k))));
      v.insert(KB(k), gtsam::imuBias::ConstantBias(out.b_accel, out.b_gyro));
      if (kp != k) {
        v.insert(KX(kp), I.sm->calculateEstimate<gtsam::Pose3>(KX(kp)));
        v.insert(KV(kp), I.sm->calculateEstimate<gtsam::Vector3>(KV(kp)));
        v.insert(KB(kp),
                 I.sm->calculateEstimate<gtsam::imuBias::ConstantBias>(KB(kp)));
      }
      if (I.last_imu) out.resid_imu = I.last_imu->error(v);
      if (I.last_reg) out.resid_reg = I.last_reg->error(v);
    } catch (const std::exception& e) {
      out.resid_imu = out.resid_reg = -1.0;
    }
  }

  // ---- the in-window keyframes -------------------------------------------
  // A keyframe IS a scan, so there is no new variable: "moving a keyframe" is
  // reading X(k_j) for a keyframe scan still inside the window. FREEZE ON
  // MARGINALISATION: once the key has left the window it never moves again.
  if (out.valid && P.keyframe_writeback) {
    const auto& stamps = I.sm->timestamps();
    for (auto it = I.kf_key.begin(); it != I.kf_key.end();) {
      const long key = it->second;
      if (stamps.find(KX(key)) == stamps.end()) {
        I.kf_frozen_upto = std::max(I.kf_frozen_upto, key);
        out.kf_frozen.push_back(it->first);
        it = I.kf_key.erase(it);          // frozen forever; stop tracking it
        continue;
      }
      if (key != k) {
        try {
          out.kf_poses.emplace_back(
              it->first,
              Eigen::Matrix4d(out.anchor_gauge *
                              mat_of(I.sm->calculateEstimate<gtsam::Pose3>(KX(key)))));
        } catch (const std::exception& e) {
          // a key inside the window that cannot be read is a defect, not a skip
          out.update_exception = true;
        }
      }
      ++it;
    }
  }

  // ---- prune the re-anchor's side maps to the window ----------------------
  // O(window), not O(run): a key that has left the window can never be
  // anchored again and its DLIO pose is never read again. k and k-1 are kept
  // unconditionally -- k-1 is what the increment needs and it may already have
  // been marginalised on a very short lag.
  {
    const auto& stamps = I.sm->timestamps();
    for (auto it = I.key_time.begin(); it != I.key_time.end();) {
      if (it->first >= k - 1 || stamps.find(KX(it->first)) != stamps.end()) {
        ++it;
      } else {
        I.dlio_pose.erase(it->first);
        it = I.key_time.erase(it);
      }
    }
  }

  out.reseats = I.reseats;
  out.singular_info = I.singular_info;
  out.window_factors = (int)I.sm->getFactors().size();
  if (out.window_vars == 0) out.window_vars = (int)I.sm->timestamps().size();
  out.solve_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_begin).count();
  return out;
}

}  // namespace smoother
}  // namespace dlio
