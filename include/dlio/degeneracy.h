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
#include <algorithm>
#include <cmath>

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

}  // namespace degeneracy
}  // namespace dlio

#endif  // DLIO_DEGENERACY_H
