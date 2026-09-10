/***********************************************************
 *                                                         *
 * slamlab addition — GICP degeneracy guard                *
 *                                                         *
 * Pure Eigen. NO ros, NO pcl, NO dlio headers, on purpose: *
 * it is unit-testable with a bare g++ -I/usr/include/eigen3 *
 * inside the recipe image (see test_degeneracy.cpp).       *
 *                                                         *
 ***********************************************************/

#ifndef DLIO_DEGENERACY_H
#define DLIO_DEGENERACY_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace dlio {
namespace degeneracy {

// Knobs, all read from ~dlio/odom/gicp/degeneracy/* in OdomNode::getParams().
// EVERY default reproduces stock DLIO behaviour: the guard is off and scores
// nothing until someone asks for it.
struct Params {
  bool   enabled       = false;  // act: project the observer error onto the observable subspace
  bool   observe       = false;  // score + log only, zero arithmetic change (implied by enabled)
  double min_ratio     = 0.02;   // tau_lo: lambda_k/lambda_max below this => w=0 (unobservable)
  double full_ratio    = 0.10;   // tau_hi: above this => w=1; smoothstep in between
  // ABSOLUTE per-point information, q_k = lambda_k / ncorr, tested BESIDE the
  // ratio and combined by taking the TIGHTER of the two weights.
  //
  // Why a second measure at all. Htt = 0.5*N*I + 499.5*sum n_i n_i^T under PLANE
  // regularisation, so q_k = lambda_k/N is - up to those two constants - the MEAN
  // SQUARED COMPONENT of the correspondence normals along direction k: count-free
  // and scene-intrinsic, in a fixed range, where the raw eigenvalue is not. The
  // RATIO lambda_k/lambda_max is count-free too, but it is blind to the failure
  // that actually happened on the big Sandland bag (ad3517ba, aa/AA_ANALYSIS.md
  // section 4): when the submap goes STALE the whole Hessian collapses, lambda_max
  // falls WITH lambda_min, and the ratio stays at the bag median while the scan
  // carries an order of magnitude less information than a healthy draw of the same
  // seconds. Over bag t 355-367 the diverging draw's q_min was 1.3-2.4 against
  // 5.2-31 in each of the two surviving draws AT THE SAME correspondence count;
  // its r_min, 0.08-0.21, was unremarkable. One measure sees a THIN scene, the
  // other sees a scene that has stopped being measured; a corridor can do both.
  //
  // full_info = 0 disables the test entirely (the shipped default), so an existing
  // row that sets neither key keeps exactly the arithmetic it had.
  double min_info      = 0.0;    // q_lo: lambda_k/ncorr below this => w=0 before the floor
  double full_info     = 0.0;    // q_hi: above this => w=1. 0 => the info test is OFF
  int    max_weak_dirs = 1;      // never down-weight more than this many of the 3 directions
  // FLOOR on the observability weight. w = 0 does not merely ignore a bad
  // measurement: it disconnects the axis from every correction updateState()
  // makes, so state.p, state.v AND state.b.accel receive nothing along u_min and
  // that axis becomes a FREE DOUBLE INTEGRATOR for as long as the direction stays
  // weak. On the big Sandland corridor that is ~780 s: a 0.05-0.1 deg residual
  // attitude error leaks 0.009-0.017 m/s^2 of gravity into the axis, and
  // 0.5*a*t^2 is 250 m to 5 km of drift (|v| reaching 7-13 m/s). Zhang/Kaess/Singh
  // remapping gets away with w = 0 because a LOAM degeneracy lasts seconds.
  // innov_max_m does NOT bound this: it clamps the magnitude of err, not the
  // integrated drift. Only a PARTIAL weight can produce the bounded linear drift
  // this guard claims as its success state, so the weakest direction keeps
  // min_weight of its authority. Set 0.0 only for a deliberate remapping arm.
  double min_weight    = 0.25;   // w_min floor for the down-weighted directions
  int    min_corr      = 200;    // fewer correspondences than this => refuse to judge (w=1)
  bool   guard_pose    = false;  // also guard the GICP translation increment (T / T_corr)
  double innov_max_m   = 0.0;    // 0 = off; else clamp |err| to this many metres
  int    log_every     = 1;      // scans between [DEGEN] lines

  bool scoring() const { return enabled || observe; }
  // The absolute test is opt-in: a band of 0/0 leaves the ratio test alone.
  bool info_test() const { return full_info > 0.0; }
};

// One scan's verdict.
struct Weights {
  bool            valid  = false;                          // Hessian fresh AND enough correspondences
  Eigen::Matrix3d U      = Eigen::Matrix3d::Identity();    // columns = eigenvectors, ASCENDING eigenvalue
  Eigen::Vector3d lambda = Eigen::Vector3d::Ones();        // eigenvalues, ascending
  Eigen::Vector3d ratio  = Eigen::Vector3d::Ones();        // lambda_k / lambda_max, in (0,1]
  Eigen::Vector3d info   = Eigen::Vector3d::Ones();        // lambda_k / ncorr (absolute, count-free)
  Eigen::Vector3d w      = Eigen::Vector3d::Ones();        // per-direction observability weight
  int             ncorr  = 0;
  int             weak   = 0;                              // how many directions have w < 1

  double lambda_min() const { return lambda(0); }
  double lambda_max() const { return lambda(2); }
  double ratio_min()  const { return ratio(0);  }
  double info_min()   const { return info(0);   }
  double w_min()      const { return w.minCoeff(); }
  bool   degenerate() const { return weak > 0; }
  // True when the guard would change nothing at all, whatever the error is.
  bool   is_identity() const { return w(0) == 1.0 && w(1) == 1.0 && w(2) == 1.0; }
};

// 0 below lo, 1 above hi, C1 ramp between. A hard step when the band is empty
// or inverted, so a misconfigured pair still behaves predictably.
inline double smoothstep(double r, double lo, double hi) {
  if (!(hi > lo)) return (r >= hi) ? 1.0 : 0.0;
  if (r <= lo) return 0.0;
  if (r >= hi) return 1.0;
  const double t = (r - lo) / (hi - lo);
  return t * t * (3.0 - 2.0 * t);
}

// The TRANSLATION block of a 6x6 GICP Hessian, [rot(0..2) | trans(3..5)].
//
// nano_gicp::NanoGICP::linearize builds dtdx0 with
//   dtdx0.block<3,3>(0,0) = skew(R p_A + t)   (rotation columns 0..2)
//   dtdx0.block<3,3>(0,3) = -Identity         (translation columns 3..5)
// (src/nano_gicp/nano_gicp.cc:280-281), so H = sum J^T M J puts the translation
// information at block (3,3) — and because M_i = (cov_B + T cov_A T^T)^-1 with
// BOTH clouds already in the global frame (dlio::OdomNode::getNextPose sets the
// source and target from global-frame clouds), that block is in the SAME frame
// as the observer error err = pin - state.p. Nothing needs rotating.
//
// It is a one-liner and it is still a function: the choice of block, not the
// arithmetic, is the part that can be silently wrong, and here a test can pin it.
inline Eigen::Matrix3d translation_information(const Eigen::Matrix<double, 6, 6>& H) {
  return H.block<3, 3>(3, 3);
}

// The ROTATION block of the same 6x6, and the record that goes with it.
//
// EVERY lambda/r/w/u this programme has published so far is TRANSLATION:
// scoreDegeneracy() reads translation_information(H) and discards the rest of
// the 6x6. But the big Sandland divergence ONSET is a HEADING error -- the pose
// went 15 deg off while the free-running IMU was only 5.7 deg off, through an
// 80 deg/s turn at a bend with a 6 m sight line (SUBMAP.md section 6,
// ONSET_IMU.md verdict 2) -- and a translation-only instrument is structurally
// blind to it.
//
// This is INSTRUMENTATION ONLY. Nothing acts on it: no weight is derived from
// it, updateState() never sees it, and the guard's arithmetic is untouched. It
// exists so the rotation column can be FITTED on raw observe data before anyone
// proposes an action on it ([[fit_sigma_on_raw_only]]).
//
// UNITS WARNING, and it is why cond6 is reported with a units token. The
// rotation block's entries are (information x length^2) and the translation
// block's are (information); under PLANE regularization neither carries a
// physical scale, and the two are NOT commensurable. `cond6` -- the condition
// number of the whole 6x6 -- therefore mixes units and is a RELATIVE series
// only: comparable across scans of one run and across runs at the same
// parameters, never a physical conditioning.
inline Eigen::Matrix3d rotation_information(const Eigen::Matrix<double, 6, 6>& H) {
  return H.block<3, 3>(0, 0);
}

struct RotRecord {
  bool valid = false;
  Eigen::Vector3d lambda = Eigen::Vector3d::Ones();   // ascending
  Eigen::Vector3d u_min  = Eigen::Vector3d::UnitZ();  // weak rotation axis
  double cond6 = 1.0;                                 // full 6x6, MIXED UNITS

  double ratio_min() const {
    return (this->lambda(2) > 0.0) ? this->lambda(0) / this->lambda(2) : 1.0;
  }
};

// Read the rotation block and the full 6x6 spectrum. `hessian_valid` is
// LsqRegistration::hasFinalHessian(), for exactly the reason the translation
// scorer refuses without it: an unconverged scan leaves the PREVIOUS scan's
// Hessian in the member, and judging this scan on last scan's geometry is a
// silent no-op.
inline RotRecord rotation_record(const Eigen::Matrix<double, 6, 6>& H,
                                 bool hessian_valid) {
  RotRecord R;
  if (!hessian_valid || !H.allFinite()) { return R; }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(rotation_information(H));
  if (es.info() != Eigen::Success) { return R; }
  R.lambda = es.eigenvalues();                 // Eigen returns ascending
  R.u_min = es.eigenvectors().col(0);
  if (!(R.lambda(2) > 0.0)) { return R; }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es6(H);
  if (es6.info() == Eigen::Success) {
    const double lo = es6.eigenvalues()(0), hi = es6.eigenvalues()(5);
    R.cond6 = (lo > 0.0) ? hi / lo : std::numeric_limits<double>::infinity();
  }
  R.valid = true;
  return R;
}

// Score the translation block of the GICP Hessian.
//
// Htt = H.block<3,3>(3,3) = sum_i (cov_B + R cov_A R^T)^-1 restricted to 3x3 —
// the point-to-plane NORMAL INFORMATION matrix, already in the target/global
// frame (nano_gicp::NanoGICP::linearize uses a LEFT perturbation with the state
// ordered [rot(0..2), trans(3..5)]), i.e. the same frame as the observer error.
//
// Its ABSOLUTE scale is meaningless: regularization_method_ = PLANE forces every
// point covariance's singular values to (1, 1, 1e-3), so Htt ~ 500 * sum n_i n_i^T
// and grows with the correspondence count. Only the RATIO lambda_k/lambda_max is
// scene- and count-invariant, so that is what is thresholded.
//
// hessian_valid must come from LsqRegistration::hasFinalHessian(): final_hessian_
// is only assigned on an ACCEPTED LM step, so an unconverged scan leaves the
// PREVIOUS scan's Hessian in the member. Judging this scan on last scan's
// geometry is a silent no-op; we refuse instead.
inline Weights degeneracy_weights(const Eigen::Matrix3d& Htt,
                                  bool hessian_valid,
                                  int ncorr,
                                  const Params& p) {
  Weights W;
  W.ncorr = ncorr;

  if (!p.scoring())    return W;   // valid=false, w=(1,1,1): nothing computed, nothing applied
  if (!hessian_valid)  return W;   // stale final_hessian_ — refuse to judge
  if (ncorr < p.min_corr) return W;

  const Eigen::Matrix3d S = 0.5 * (Htt + Htt.transpose());   // symmetrise; it is PSD by construction
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S);
  if (es.info() != Eigen::Success) return W;

  W.lambda = es.eigenvalues();       // ascending
  W.U      = es.eigenvectors();      // columns match, orthonormal
  const double lmax = W.lambda(2);
  if (!(lmax > 0.0) || !std::isfinite(lmax)) return W;

  W.valid = true;
  const int maxweak = std::max(0, std::min(3, p.max_weak_dirs));
  const double wfloor = std::min(1.0, std::max(0.0, p.min_weight));
  for (int k = 0; k < 3; ++k) {
    const double lk = std::max(0.0, W.lambda(k));
    W.ratio(k) = lk / lmax;
    W.info(k)  = lk / static_cast<double>(std::max(1, ncorr));
    // Only the maxweak WEAKEST directions may be down-weighted; the eigenvalues
    // are ascending so those are exactly indices 0 .. maxweak-1. Zeroing all
    // three would hand the whole pose to dead reckoning.
    //
    // The floor is applied AFTER the ramp, so the guard can attenuate an
    // unobservable direction but never disconnect it (see Params::min_weight).
    // min_weight = 1 therefore reproduces stock DLIO exactly and reports weak=0,
    // which keeps the silent-no-op pairing honest instead of raising a false
    // alarm about a guard the operator asked to do nothing.
    //
    // Two criteria, the TIGHTER one wins, then the floor. Neither can rescue a
    // direction the other condemns: a scan is trusted along k only if it is both
    // well-CONDITIONED (ratio) and actually INFORMED (info). Taking the minimum
    // rather than, say, a product keeps each band readable on its own scale -
    // each one alone still produces exactly the weight it would have produced.
    double wk = smoothstep(W.ratio(k), p.min_ratio, p.full_ratio);
    if (p.info_test()) {
      wk = std::min(wk, smoothstep(W.info(k), p.min_info, p.full_info));
    }
    W.w(k) = (k < maxweak) ? std::max(wfloor, wk) : 1.0;
    if (W.w(k) < 1.0) ++W.weak;
  }
  return W;
}

// Project a world-frame vector onto the observable subspace: keep the component
// along each eigenvector scaled by its weight. w = (1,1,1) returns v BITWISE
// unchanged — that is what makes "guard off" an exact no-op rather than a
// numerically-close one.
inline Eigen::Vector3f project_observable(const Eigen::Vector3f& v, const Weights& W) {
  if (!W.valid || W.is_identity()) return v;
  Eigen::Vector3d vd = v.cast<double>();
  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  for (int k = 0; k < 3; ++k) out += W.w(k) * (W.U.col(k).dot(vd)) * W.U.col(k);
  return out.cast<float>();
}

// Innovation gate: a jump this big in one ~0.1 s scan is not physics, it is a bad
// registration. Clamps the MAGNITUDE (keeps the direction) rather than dropping
// the scan. max_m <= 0 disables it. Returns v unchanged when it does not bind.
inline Eigen::Vector3f clamp_innovation(const Eigen::Vector3f& v, double max_m) {
  if (!(max_m > 0.0)) return v;
  const double n = v.norm();
  if (!(n > max_m)) return v;
  return v * static_cast<float>(max_m / n);
}


// =========================================================================
//  E2 — THE TRUE 6x6, RE-ANCHORED ON THE SENSOR
//  keeper/lio_concept/dlio_smoother/E4/E2/E2.md
//
//  INSTRUMENTATION ONLY. Nothing below is read by the guard, by
//  updateState(), or by any arithmetic on the trajectory. It exists because
//  E0 could only reconstruct 5 of the 6 free parameters of each 3x3 diagonal
//  block from the eigen-summaries this record already carries, and NONE of
//  the rotation<->translation CROSS block -- and the cross block is exactly
//  what re-anchors the rotation information from the WORLD ORIGIN onto the
//  SENSOR. E0 measured the cost of not having it: the fitted attitude weight
//  s_r differed 6.62x between the big bag and Dave, and 6.85 is the ratio of
//  the squared distances from the world origin (E0.md section 3). That is a
//  lever arm, not a scene.
//
//  THE FRAMES, stated once, because getting this wrong is silent.
//
//  nano_gicp perturbs on the LEFT, in the WORLD frame, with a DECOUPLED
//  parameterisation (src/nano_gicp/lsq_registration.cc step_lm:
//      delta.linear() = so3_exp(d.head<3>());  delta.translation() = d.tail<3>();
//      x0 = delta * x0)
//  and the residual Jacobian is built from the point's WORLD coordinate
//  (src/nano_gicp/nano_gicp.cc: dtdx0.block<3,3>(0,0) = skewd(transed_mean_A),
//  dtdx0.block<3,3>(0,3) = -I). So
//
//      H_world = sum_i J_i^T M_i J_i,   J_i = [ skew(y_i) , -I ],   y_i in WORLD
//
//  is the information about xi_w = [omega_w ; tau_w] -- a rotation about the
//  WORLD ORIGIN and a world translation. Block order [rot(0..2) | trans(3..5)].
//
//  GTSAM's Pose3 retracts on the RIGHT, in the BODY frame, with the SAME
//  [rot | trans] block order: X <- X * Exp(xi_b). To first order the two
//  charts are related by the adjoint,
//
//      xi_w = Ad(T) xi_b,     Ad(T) = [[ R , 0 ], [ skew(t) R , R ]]
//
//  (gtsam::Pose3::AdjointMap, same ordering), hence
//
//      LAMBDA_body = Ad(T)^T H_world Ad(T).
//
//  AND THAT PRODUCT IS THE RE-ANCHORING. Writing y_i = t + r_i with r_i the
//  point relative to the SENSOR, the rotation block of the product collapses
//  algebraically to
//
//      LAMBDA_body(0:3,0:3) = R^T ( sum_i skew(r_i)^T M_i skew(r_i) ) R
//
//  -- the sensor-anchored attitude information, with every |t|^2 term gone.
//  The cancellation consumes H_rt and H_tr, which is precisely why no
//  eigen-summary of the two DIAGONAL blocks can do it and why E0 had to use
//  rlam as a per-run scalar. test_degeneracy.cpp case 26 asserts the identity
//  and case 27 asserts that the contamination it removes is the |t|^2 E0
//  measured.
//
//  The translation block is UNMOVED by the re-anchoring and only rotated:
//      LAMBDA_body(3:6,3:6) = R^T H_tt R
//  which is what E0's PREREG section 2 already used, and case 26 pins it too.
//
//  FIRST-ORDER, and say so: nano_gicp's chart is decoupled (so3_exp on the
//  rotation, a plain translation) and GTSAM's Exp is the SE(3) exponential.
//  The two agree at xi = 0 and differ at second order, so this conversion is
//  exact for a Hessian AT the linearisation point and nowhere else. Nobody
//  should later "fix" it into a bug.
// =========================================================================

// skew(v), as a 3x3. Named because the SIGN convention is the thing that goes
// silently wrong: this is the one for which skew(a) b == a.cross(b).
inline Eigen::Matrix3d skew3(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<     0.0, -v(2),  v(1),
         v(2),    0.0, -v(0),
        -v(1),  v(0),    0.0;
  return S;
}

// Ad(T) for T = (R, t), in [rot(0..2) | trans(3..5)] order -- the ordering
// nano_gicp and gtsam::Pose3 happen to share. Maps a RIGHT/BODY twist to the
// LEFT/WORLD twist that produces the same motion of the rigid body.
inline Eigen::Matrix<double, 6, 6> adjoint_rot_trans(const Eigen::Matrix3d& R,
                                                     const Eigen::Vector3d& t) {
  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  A.block<3, 3>(0, 0) = R;
  A.block<3, 3>(3, 0) = skew3(t) * R;
  A.block<3, 3>(3, 3) = R;
  return A;
}

// H_world (left/world, origin-anchored) -> Lambda_body (right/body,
// sensor-anchored). Symmetrised on the way out: H is a sum of J^T M J and is
// symmetric to round-off, and a factor information matrix that is not exactly
// symmetric is a source of asymmetric bugs downstream.
inline Eigen::Matrix<double, 6, 6> body_information(const Eigen::Matrix<double, 6, 6>& H,
                                                    const Eigen::Matrix3d& R,
                                                    const Eigen::Vector3d& t) {
  const Eigen::Matrix<double, 6, 6> A = adjoint_rot_trans(R, t);
  Eigen::Matrix<double, 6, 6> L = A.transpose() * H * A;
  return 0.5 * (L + L.transpose());
}

// The 21 free entries of a symmetric 6x6, ROW-MAJOR over the UPPER triangle:
//   (0,0)(0,1)...(0,5) (1,1)...(1,5) (2,2)...(2,5) (3,3)(3,4)(3,5) (4,4)(4,5) (5,5)
// One order, written down once, used by the printf and by the reader.
inline void upper21(const Eigen::Matrix<double, 6, 6>& M, double out[21]) {
  int k = 0;
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) out[k++] = M(i, j);
}

// The inverse of upper21 -- so a test can round-trip and a reader has one
// authority for the order rather than two transcriptions of it.
inline Eigen::Matrix<double, 6, 6> from_upper21(const double in[21]) {
  Eigen::Matrix<double, 6, 6> M;
  int k = 0;
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) { M(i, j) = in[k]; M(j, i) = in[k]; ++k; }
  return M;
}

// SO(3) log: the rotation VECTOR (axis * angle, radians) of R.
//
// Via the QUATERNION, not via acos((tr-1)/2), and the reason is measured. The
// trace route is ill-conditioned in a window around pi far wider than it looks:
// at theta = pi - 2.7e-6 it returns |w| = 3.1416034 for a true 3.14159, an
// error of 1.3e-5 rad, because the scale factor 0.5*theta/sin(theta) inherits
// the relative error of a theta recovered from a trace that has gone flat.
// Eigen's matrix-to-quaternion picks its branch from the largest diagonal term
// and stays conditioned everywhere, and atan2(|v|, w) is conditioned at both
// ends. Round-trips to 1e-13 or better from 1e-9 rad to pi.
inline Eigen::Vector3d log_so3(const Eigen::Matrix3d& R) {
  Eigen::Quaterniond q(R);
  q.normalize();
  if (q.w() < 0.0) q.coeffs() *= -1.0;      // shortest path: theta in [0, pi]
  const double n = q.vec().norm();
  if (n < 1e-12) {                           // theta -> 0
    if (q.w() == 0.0) return Eigen::Vector3d::Zero();
    return Eigen::Vector3d((2.0 / q.w()) * q.vec());
  }
  return Eigen::Vector3d((2.0 * std::atan2(n, q.w()) / n) * q.vec());
}

// The DECOUPLED relative-pose 6-vector, [rotvec(3) ; translation(3)], for a
// relative pose A^-1 B:
//     v(0:3) = log_so3( R_A^T R_B )              radians
//     v(3:6) = R_A^T ( t_B - t_A )               metres, in A's own frame
// Rebuild EXACTLY with gtsam.Pose3(gtsam.Rot3.Expmap(v[0:3]), v[3:6]).
// This is NOT gtsam's Pose3::Logmap, which puts V(omega)^-1 on the
// translation; the decoupled form is chosen because it is invertible offline
// with no extra machinery and because reading it needs no convention lookup.
inline Eigen::Matrix<double, 6, 1> relative_rot_trans(const Eigen::Matrix3d& RA,
                                                      const Eigen::Vector3d& tA,
                                                      const Eigen::Matrix3d& RB,
                                                      const Eigen::Vector3d& tB) {
  Eigen::Matrix<double, 6, 1> v;
  v.head<3>() = log_so3(RA.transpose() * RB);
  v.tail<3>() = RA.transpose() * (tB - tA);
  return v;
}

// One scan's E2 record. Everything here is derived from the SAME
// getFinalHessian() the two eigen-summary blocks are derived from, at the same
// linearisation point, so a disagreement between them is a bug and a reader
// can check for one.
struct HessianRecord {
  bool valid = false;
  double Hw[21] = {0};      // WORLD/LEFT, origin-anchored -- as nano_gicp built it
  double Hb[21] = {0};      // BODY/RIGHT, sensor-anchored -- Ad^T H Ad
  double ferr = 0.0;        // getFinalError(): sum e^T M e at the accepted step
  int    ncorr = 0;         // gicp.num_correspondences, UNGATED
};

inline HessianRecord hessian_record(const Eigen::Matrix<double, 6, 6>& H,
                                    bool hessian_valid,
                                    const Eigen::Matrix3d& R,
                                    const Eigen::Vector3d& t,
                                    double final_error, int ncorr) {
  HessianRecord Q;
  Q.ferr = final_error;
  Q.ncorr = ncorr;
  // Same refusal as the two scorers, for the same reason: final_hessian_ is
  // only written on an ACCEPTED LM step, so an unconverged scan still holds the
  // PREVIOUS scan's geometry and dumping it would put one scan's matrix on
  // another scan's line.
  if (!hessian_valid || !H.allFinite()) return Q;
  const Eigen::Matrix<double, 6, 6> L = body_information(H, R, t);
  if (!L.allFinite()) return Q;
  upper21(H, Q.Hw);
  upper21(L, Q.Hb);
  Q.valid = true;
  return Q;
}

}  // namespace degeneracy
}  // namespace dlio

#endif  // DLIO_DEGENERACY_H
