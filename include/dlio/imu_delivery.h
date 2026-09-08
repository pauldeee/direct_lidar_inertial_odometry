/***********************************************************
 *  slamlab: what DLIO could not see about its own IMU.
 ***********************************************************/

#ifndef DLIO_IMU_DELIVERY_H
#define DLIO_IMU_DELIVERY_H

namespace dlio {
namespace imu_delivery {

// THE BANNER WAS STRUCTURALLY BLIND, and this is the statistic that is not.
//
// odom.cc pushes 1/dt per received sample and the status banner prints the mean
// of the last window (`avg_imu_rate`). On the big Sandland bag ad3517ba that
// read 632.44 Hz while the node was in fact being handed 471.95 Hz of a
// 640.0086 Hz stream: 98.87 % of the delivered intervals were one clean
// 1.5625 ms tick and a mean of 1/dt is dominated by them, so 2,048 gaps over
// 50 ms, a 637 ms worst gap and 298 s of blackout in a 1,140 s bag left the
// printed rate 0.1 % away from a perfect stream. (Whole messages were being
// discarded upstream, by the replay player's publisher queue; the fix is there.
// This is the witness that says whether it worked.)
//
// COUNT OVER SPAN cannot be fooled the same way: (n-1) intervals divided by
// (last - first) IS the delivered rate, and it falls in exact proportion to
// whatever went missing, whatever shape the surviving intervals have. The worst
// interval and the number of intervals over a stated threshold then say where
// the loss sat, which a rate alone never can.
//
// Header-only and dependency-free on purpose: it is compiled and RUN by the
// image build (test_degeneracy.cpp), so an arithmetic mistake here fails the
// build instead of a 40-minute bag replay.
struct Counter {

  long   n         = 0;      // messages received
  double first     = 0.;     // stamp of the first, seconds
  double last      = 0.;     // stamp of the most recent, seconds
  double max_dt    = 0.;     // longest interval between two received samples, s
  double max_dt_at = 0.;     // ... the stamp that closed it
  long   gaps      = 0;      // intervals longer than gap_s
  double gap_s     = 0.05;   // what counts as a gap (IMU_CHECK.md counts 50 ms)

  // Record one received sample. Returns true when THIS sample closed an
  // interval longer than gap_s, so a caller can say so once, loudly.
  //
  // A non-advancing stamp (duplicate, or out of order across a reconnect) is
  // COUNTED but contributes no interval: it is not evidence of a gap, and
  // letting it produce a negative dt would silently poison max_dt.
  bool note(double stamp) {
    if (n == 0) {
      first = stamp;
      last = stamp;
      n = 1;
      return false;
    }
    ++n;
    const double dt = stamp - last;
    if (dt <= 0.) return false;      // duplicate / out of order: no interval
    last = stamp;
    bool gap = false;
    if (dt > gap_s) { ++gaps; gap = true; }
    if (dt > max_dt) { max_dt = dt; max_dt_at = stamp; }
    return gap;
  }

  double span() const { return (n > 1 && last > first) ? last - first : 0.; }

  // The DELIVERED rate. 0 until there are two samples spanning real time --
  // never a fabricated number from one sample.
  double hz() const {
    const double s = span();
    return s > 0. ? (double)(n - 1) / s : 0.;
  }

  // What fraction of an expected rate actually arrived. 0 expected => 0, so a
  // caller that does not know the nominal gets a number it cannot misread.
  double delivered_fraction(double expected_hz) const {
    return expected_hz > 0. ? hz() / expected_hz : 0.;
  }

};

}  // namespace imu_delivery
}  // namespace dlio

#endif
