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
  int    max_weak_dirs = 1;      // never down-weight more than this many of the 3 directions
  int    min_corr      = 200;    // fewer correspondences than this => refuse to judge (w=1)
  bool   guard_pose    = false;  // also guard the GICP translation increment (T / T_corr)
  double innov_max_m   = 0.0;    // 0 = off; else clamp |err| to this many metres
  int    log_every     = 1;      // scans between [DEGEN] lines

  bool scoring() const { return enabled || observe; }
};

// One scan's verdict.
struct Weights {
  bool            valid  = false;                          // Hessian fresh AND enough correspondences
  Eigen::Matrix3d U      = Eigen::Matrix3d::Identity();    // columns = eigenvectors, ASCENDING eigenvalue
  Eigen::Vector3d lambda = Eigen::Vector3d::Ones();        // eigenvalues, ascending
  Eigen::Vector3d ratio  = Eigen::Vector3d::Ones();        // lambda_k / lambda_max, in (0,1]
  Eigen::Vector3d w      = Eigen::Vector3d::Ones();        // per-direction observability weight
  int             ncorr  = 0;
  int             weak   = 0;                              // how many directions have w < 1

  double lambda_min() const { return lambda(0); }
  double lambda_max() const { return lambda(2); }
  double ratio_min()  const { return ratio(0);  }
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
  for (int k = 0; k < 3; ++k) {
    const double lk = std::max(0.0, W.lambda(k));
    W.ratio(k) = lk / lmax;
    // Only the maxweak WEAKEST directions may be down-weighted; the eigenvalues
    // are ascending so those are exactly indices 0 .. maxweak-1. Zeroing all
    // three would hand the whole pose to dead reckoning.
    W.w(k) = (k < maxweak) ? smoothstep(W.ratio(k), p.min_ratio, p.full_ratio) : 1.0;
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
