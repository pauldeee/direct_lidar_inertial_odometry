/***********************************************************
 *  INCREMENT 1 — THE THREE REPAIRS (slamlab)
 *
 *  keeper/lio_concept/CONCEPT.md section 4, INCREMENT 1.
 *  keeper/sandland_2860/big_bag/submap_check/SUBMAP.md sections 1, 5, 6.
 *
 *  Three defects of stock DLIO, each isolated here as a PURE function so it can
 *  be unit-tested without ROS, without PCL and without a 40-minute bag replay
 *  (the same discipline as dlio/degeneracy.h and dlio/imu_delivery.h):
 *
 *    (1a-i)  pushSubmapIndices() takes the WHOLE candidate list, at ANY
 *            distance, whenever that list is shorter than k. On the big
 *            Sandland bag the concave hull returns 4-8 vertices against
 *            kcc = 10, so keyframes 0-3 -- 68 m away, in the opening chamber --
 *            are in EVERY submap the run builds.  refuse_short_candidate_list()
 *            says when that has happened.
 *
 *    (1b)    updateKeyframes() measures ONE number, the Euclidean distance to
 *            the closest keyframe OF ANY AGE, and uses it to answer "does the
 *            submap already cover what I can see?".  On a first traverse the
 *            two coincide; on a REVISIT they do not, and the run stops making
 *            keyframes entirely (28.80 s, 15.77 m and 162 deg of believed
 *            motion with no keyframe at all).  age_clause() is the added
 *            disjunct, OFF by default.
 *
 *    (N57)   geo/abias_max is an ABSOLUTE constant sitting BELOW this rig's own
 *            turn-on bias (0.3 against an Octagon init of 0.305), so the clamp
 *            clips the CALIBRATED value.  derive_abias_clamp() replaces the
 *            constant with the run's own 3 s init calibration plus a margin.
 *
 *  Every default in here reproduces stock DLIO exactly:
 *    max_age_s   = 0  -> age_clause() is always false
 *    margin      = 0  -> derive_abias_clamp() returns the configured constant
 *  A row that sets nothing gets a byte-identical parameter dump and a
 *  byte-identical arithmetic path.
 ***********************************************************/

#ifndef DLIO_REPAIRS_H_
#define DLIO_REPAIRS_H_

#include <Eigen/Core>

#include <cmath>
#include <cstddef>

namespace dlio {
namespace repairs {

// ---------------------------------------------------------------------------
// (1a-i) THE UNBOUNDED CANDIDATE LIST
// ---------------------------------------------------------------------------
//
// pushSubmapIndices(dists, k, frames) promises "every candidate whose distance
// is <= the k-th smallest".  With fewer than k candidates there IS no k-th
// smallest, and stock DLIO's max-heap then yields the LARGEST element it holds,
// so the inclusive test admits everything -- at any distance.
//
// Whether that is a defect depends on what the candidate list IS:
//
//   * the knn call passes the WHOLE keyframe population.  "The 20 nearest of
//     the 3 that exist" IS all 3; there is no nearer keyframe being passed over,
//     and refusing would leave the run with no submap at all in its first
//     seconds.  Stock behaviour is correct and is kept, verbatim.
//
//   * the convex- and concave-hull calls pass a FILTERED SUBSET.  Here "shorter
//     than k" says nothing about how far away the members are, and there ARE
//     nearer keyframes outside the subset.  Taking the whole hull at unbounded
//     distance is not "the k nearest of the hull"; it is "the hull", which is a
//     different set, and it is how four keyframes 68 m outside the mapped
//     corridor entered every submap on ad3517ba.
//
// So the refusal is conditioned on the list being a proper subset, not on the
// count alone.  `population` is the number of keyframes the distances were
// computed over; `n` is the length of the candidate list actually passed.
//
// WHY THE REFUSAL AND NOT A NEW ALPHA.  The other available repair is to derive
// the concave hull's alpha from the measured keyframe spacing instead of
// leaving it at keyframe_thresh_dist_.  It is rejected here for three reasons:
//   (i)  it is a FIFTH hand-fitted band on a programme that has had four fail;
//        alpha would have to be fitted per scene, and the alpha-complex is a
//        step function of it (of 1,088 Delaunay tetrahedra over RUN 2's 166
//        keyframes, 2 have circumradius < 1.0 m -- nothing continuous to tune);
//   (ii) it CHANGES the hull membership, so it is a behavioural change to the
//        submap in every scene, not a repair -- it cannot ship in a rung whose
//        whole claim is inertness;
//   (iii) the refusal is a property of pushSubmapIndices itself and therefore
//        protects the convex call too, which the alpha fix would not touch.
// With the refusal the kcc term contributes nothing on this bag.  That is the
// honest outcome: the term was never doing what it was designed to do, and
// making it do something is a hypothesis for another increment.
inline bool refuse_short_candidate_list(std::size_t n, int k, std::size_t population) {
  if (k <= 0) { return false; }                    // k <= 0 is not a bound
  if (n >= (std::size_t)k) { return false; }       // a genuine k-th smallest exists
  return n < population;                           // a SUBSET, short of k -> refuse
}

// ---------------------------------------------------------------------------
// (1b) THE KEYFRAME AGE CLAUSE
// ---------------------------------------------------------------------------
struct KeyframeAge {
  // Seconds.  0 (the default) = OFF: age_clause() is then false for every
  // input and updateKeyframes() runs exactly the arithmetic it always did.
  double max_age_s = 0.0;
  // Metres the pose must have travelled since MY OWN LAST keyframe before the
  // age clause may fire.  Negative (the default) = derive as 0.25 * threshD.
  double min_travel_m = -1.0;

  bool on() const { return this->max_age_s > 0.0; }

  // Resolved once, at parameter-read time, so the number that ran is the number
  // that is logged.  A caller that set min_travel_m explicitly keeps it.
  double travel_floor(double keyframe_thresh_dist) const {
    return (this->min_travel_m >= 0.0) ? this->min_travel_m
                                       : 0.25 * keyframe_thresh_dist;
  }
};

// age_s     : how long ago the CLOSEST keyframe was laid down (seconds).
// travel_m  : |state.p - last keyframe laid|, metres.  NOT the distance to the
//             closest keyframe: on a revisit those are wildly different, and it
//             is the second one that says whether this scan carries anything
//             new.
// floor_m   : the resolved travel_floor().
//
// Both conditions are required.  Age alone would lay a keyframe on a stationary
// rig forever (151 pauses on this rig's own bags); travel alone is threshD,
// which is the clause that already exists.
inline bool age_clause(double age_s, double travel_m,
                       const KeyframeAge& p, double floor_m) {
  if (!p.on()) { return false; }
  if (!(age_s > p.max_age_s)) { return false; }
  return travel_m >= floor_m;
}

// ---------------------------------------------------------------------------
// (N57) THE ACCEL-BIAS CLAMP, FROM THE RUN'S OWN CALIBRATION
// ---------------------------------------------------------------------------
//
// updateState() clamps the accel-bias state componentwise at +/- abias_max.
// Shipped rows carry 0.3, which is BELOW rig 2860's own turn-on bias (Octagon
// init 0.305; big bag 0.236-0.251), so the clamp clips a value the sensor
// really has -- and a bias railed at 0.300 against a true 0.067 over-corrects by
// 0.233 m/s^2, which is 466 mm in 2 s.
//
// The replacement is not another constant.  DLIO already measures this rig's
// bias, every run, in its own first 3 seconds, to about a millimetre per second
// squared.  So the bound becomes that measurement plus a margin, PER AXIS,
// because the bias is a per-axis sensor property and the failure was per-axis
// (it is always ab_z that rails).
//
// margin <= 0 (the default) returns the configured constant on all three axes,
// which is stock behaviour bit for bit.
inline Eigen::Vector3f derive_abias_clamp(const Eigen::Vector3f& b_init,
                                          double margin, double stock_max) {
  Eigen::Vector3f c;
  if (!(margin > 0.0)) {
    c.setConstant((float)stock_max);
    return c;
  }
  for (int i = 0; i < 3; ++i) {
    c[i] = (float)(std::fabs((double)b_init[i]) + margin);
  }
  return c;
}

}  // namespace repairs
}  // namespace dlio

#endif  // DLIO_REPAIRS_H_
