// E4 / ARCHITECT A -- THE OFFLINE A/A HARNESS.
//
// Replays a recorded draw's OWN measurements through the SAME dlio::smoother::
// Smoother the node runs, so the in-DLIO smoother can be required to reproduce
// the offline refit (E0 / E2) before any node time is spent. The claim being
// tested is not "the smoother helps" -- E0 already measured that on recorded
// data -- it is "the C++ in the node is the same estimator as the python that
// was judged". A mismatch here is a BUG, not a result.
//
// Nothing in this file parses a [DEGEN] line or joins a pose chain. The join is
// the one thing that could differ between the two sides for a reason that is
// not the estimator, so it is done ONCE, in python, by e0_lib.join_draw /
// e2_lib.join_draw6 VERBATIM (E4/IMPL/make_tape.py), and both sides consume the
// same flat tape. The bracket information model (E0's S1/S3 reconstruction) is
// likewise built in python and carried in the tape: there is exactly ONE
// information route in C++ and it is the shipping one, H6b -> Lambda.
//
//   usage: dlio_smoother_replay <tape> <params> <out_poses.ndjson> <out_stats.ndjson>
//
// The params file is `key value` per line; see E4/IMPL/impl_aa.py for the
// writer and IMPL.md for the list.

#include "dlio/smoother.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TapeRow {
  double stamp = 0, prev_stamp = 0;
  double T[16] = {0};
  double v[3] = {0};
  double H6b[21] = {0};
  double hvalid = 0, ncorr = 0, ferr = 0;
  double dp6[6] = {0};
  double dp6valid = 0;
  double lam21[21] = {0};
  double imu_lo = 0, imu_hi = 0;
};

struct Tape {
  std::vector<std::array<double, 7>> imu;
  std::vector<TapeRow> rows;
  bool has_lambda = false;
};

bool read_tape(const std::string& path, Tape* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "DLIOTAPE", 8) != 0) return false;
  std::uint32_t ver = 0, n_scans = 0, n_imu = 0, flags = 0;
  f.read((char*)&ver, 4);
  f.read((char*)&n_scans, 4);
  f.read((char*)&n_imu, 4);
  f.read((char*)&flags, 4);
  if (ver != 1u) return false;
  out->has_lambda = (flags & 1u) != 0u;
  out->imu.resize(n_imu);
  for (std::uint32_t i = 0; i < n_imu; ++i)
    f.read((char*)out->imu[i].data(), 7 * sizeof(double));
  out->rows.resize(n_scans);
  for (std::uint32_t i = 0; i < n_scans; ++i) {
    TapeRow& r = out->rows[i];
    f.read((char*)&r.stamp, sizeof(double));
    f.read((char*)&r.prev_stamp, sizeof(double));
    f.read((char*)r.T, 16 * sizeof(double));
    f.read((char*)r.v, 3 * sizeof(double));
    f.read((char*)r.H6b, 21 * sizeof(double));
    f.read((char*)&r.hvalid, sizeof(double));
    f.read((char*)&r.ncorr, sizeof(double));
    f.read((char*)&r.ferr, sizeof(double));
    f.read((char*)r.dp6, 6 * sizeof(double));
    f.read((char*)&r.dp6valid, sizeof(double));
    if (out->has_lambda) f.read((char*)r.lam21, 21 * sizeof(double));
    f.read((char*)&r.imu_lo, sizeof(double));
    f.read((char*)&r.imu_hi, sizeof(double));
  }
  return (bool)f;
}

std::map<std::string, double> read_params_num(const std::string& path,
                                              std::map<std::string, std::string>* strs) {
  std::map<std::string, double> m;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string k, v;
    ss >> k >> v;
    if (k.empty() || v.empty()) continue;
    (*strs)[k] = v;
    try { m[k] = std::stod(v); } catch (...) {}
  }
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <tape> <params> <out_poses.ndjson> <out_stats.ndjson>\n",
                 argv[0]);
    return 2;
  }
  Tape tape;
  if (!read_tape(argv[1], &tape)) {
    std::fprintf(stderr, "[REPLAY][ERROR] cannot read tape %s\n", argv[1]);
    return 3;
  }
  std::map<std::string, std::string> strs;
  std::map<std::string, double> pv = read_params_num(argv[2], &strs);
  auto get = [&](const char* k, double d) {
    auto it = pv.find(k);
    return it == pv.end() ? d : it->second;
  };

  dlio::smoother::Params P;
  P.enabled = true;
  P.lag_s = get("lag_s", P.lag_s);
  P.alpha = get("alpha", P.alpha);
  P.info_scale_trans = get("info_scale_trans", P.info_scale_trans);
  P.info_scale_rot = get("info_scale_rot", P.info_scale_rot);
  P.huber_k = get("huber_k", P.huber_k);
  P.floor_sigma_trans_m = get("floor_sigma_trans_m", P.floor_sigma_trans_m);
  P.floor_sigma_rot_deg = get("floor_sigma_rot_deg", P.floor_sigma_rot_deg);
  P.gravity = get("gravity", P.gravity);
  P.lm_max_iterations = (int)get("lm_max_iterations", P.lm_max_iterations);
  P.lm_relative_error_tol = get("lm_relative_error_tol", P.lm_relative_error_tol);
  P.marginalize_every = (int)get("marginalize_every", P.marginalize_every);
  P.gauge_sigma_rp_deg = get("gauge_sigma_rp_deg", P.gauge_sigma_rp_deg);
  P.gauge_sigma_yaw_deg = get("gauge_sigma_yaw_deg", P.gauge_sigma_yaw_deg);
  P.gauge_sigma_pos_m = get("gauge_sigma_pos_m", P.gauge_sigma_pos_m);
  // E4 INCREMENT 2. THE REPLAY DEFAULTS THESE OFF, and that is deliberate: this
  // binary's job is the offline A/A against E0's python refit (IMPL.md sec 6.4),
  // and E0's graph has the 10 m gauge and no re-anchor. Turning either on here
  // would silently change the trajectory the A/A compares. They are settable so
  // the SAME recorded tape can be replayed both ways and the standing offset
  // measured on it, at zero node cost.
  P.reanchor = get("reanchor", 0.0) != 0.0;
  P.blend_increment = get("blend_increment", 0.0) != 0.0;
  P.anchor_sigma_pos_m = get("anchor_sigma_pos_m", P.anchor_sigma_pos_m);
  P.anchor_sigma_rot_deg = get("anchor_sigma_rot_deg", P.anchor_sigma_rot_deg);
  P.standing_offset_max_m = get("standing_offset_max_m", P.standing_offset_max_m);
  P.v0_sigma = get("v0_sigma", P.v0_sigma);
  P.bias_prior_sigma_accel = get("bias_prior_sigma_accel", P.bias_prior_sigma_accel);
  P.bias_prior_sigma_gyro = get("bias_prior_sigma_gyro", P.bias_prior_sigma_gyro);
  P.bias_init_acc = get("bias_init_acc", P.bias_init_acc);
  P.bias_init_gyr = get("bias_init_gyr", P.bias_init_gyr);
  P.integration_sigma = get("integration_sigma", P.integration_sigma);
  P.keyframe_writeback = false;   // no map here: the replay tests the estimator
  for (int i = 0; i < 3; ++i) {
    char k[32];
    std::snprintf(k, sizeof(k), "vrw%d", i); P.vrw[i] = get(k, P.vrw[i]);
    std::snprintf(k, sizeof(k), "arw%d", i); P.arw[i] = get(k, P.arw[i]);
    std::snprintf(k, sizeof(k), "acc_adev%d", i);
    P.acc_adev_2s[i] = get(k, P.acc_adev_2s[i]);
    std::snprintf(k, sizeof(k), "gyr_adev%d", i);
    P.gyr_adev_2s[i] = get(k, P.gyr_adev_2s[i]);
  }
  // "h6b" (the SHIPPING route: Lambda = s * H6b + floor, built in C++) or
  // "tape" (a Lambda built in python -- E0's bracket, which exists nowhere in
  // the live path and must not).
  if (strs.count("linear_solver")) P.linear_solver = strs["linear_solver"];
  const std::string info_src = strs.count("info_source") ? strs["info_source"] : "h6b";
  if (info_src != "tape" && info_src != "h6b") {
    // A value nobody recognises must STOP, not fall through to the other
    // source: a quoted 'tape' once fell through to h6b, every registration
    // factor got ZERO information, and the replay produced a pure inertial
    // dead-reckoning that looked like a trajectory.
    std::fprintf(stderr,
                 "[REPLAY][ERROR] info_source=%s is not 'h6b' or 'tape'\n",
                 info_src.c_str());
    return 5;
  }
  if (info_src == "tape" && !tape.has_lambda) {
    std::fprintf(stderr, "[REPLAY][ERROR] info_source=tape but the tape carries "
                         "no Lambda\n");
    return 4;
  }

  Eigen::Matrix3d R_bi;
  R_bi << get("R_bi00", -1), get("R_bi01", 0), get("R_bi02", 0),
      get("R_bi10", 0), get("R_bi11", -1), get("R_bi12", 0),
      get("R_bi20", 0), get("R_bi21", 0), get("R_bi22", 1);
  Eigen::Vector3d t_bi(get("t_bi0", 0.002441), get("t_bi1", 0.009725),
                       get("t_bi2", -0.030662));

  dlio::smoother::Smoother sm(P);
  sm.setExtrinsic(R_bi, t_bi);
  sm.setBiasPrior(Eigen::Vector3d(get("b_acc0", 0), get("b_acc1", 0), get("b_acc2", 0)),
                  Eigen::Vector3d(get("b_gyr0", 0), get("b_gyr1", 0), get("b_gyr2", 0)));

  std::ofstream fp(argv[3]);
  std::ofstream fs(argv[4]);
  fp.precision(17);
  fs.precision(10);

  // TWO trajectories, because they are two different quantities and comparing
  // the wrong one would be an A/A that could not fail:
  //   lag0   what the LIVE nudge writes back -- the estimate of scan k that
  //          exists at scan k, and the only one that can exist there.
  //   final  what the offline refits publish -- the LAST estimate of scan k
  //          before it left the window, i.e. the smoothed trajectory.
  std::vector<Eigen::Matrix4d> final_T(tape.rows.size(), Eigen::Matrix4d::Identity());
  std::vector<Eigen::Vector3d> final_v(tape.rows.size(), Eigen::Vector3d::Zero());
  std::vector<char> final_seen(tape.rows.size(), 0);
  std::vector<long> wk;
  std::vector<Eigen::Matrix4d> wp;
  std::vector<Eigen::Vector3d> wv;

  dlio::smoother::NoOpLedger ledger;
  double solve_ms_sum = 0.0, solve_ms_max = 0.0;
  std::vector<double> solve_ms;
  solve_ms.reserve(tape.rows.size());
  long n_reg = 0, n_floor = 0, n_psd = 0, n_gap = 0;

  for (std::size_t i = 0; i < tape.rows.size(); ++i) {
    const TapeRow& r = tape.rows[i];
    dlio::smoother::ScanInput in;
    in.stamp = r.stamp;
    in.prev_stamp = r.prev_stamp;
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) in.T_gicp(a, b) = r.T[a * 4 + b];
    in.v_world << r.v[0], r.v[1], r.v[2];
    in.h_valid = r.hvalid != 0.0;
    in.ncorr = (int)r.ncorr;
    in.ferr = r.ferr;
    for (int a = 0; a < 6; ++a) in.dp6(a) = r.dp6[a];
    in.dp6_valid = r.dp6valid != 0.0;
    if (info_src == "tape") {
      // The tape's Lambda is FINAL (python already added the same constant
      // floor), so it is handed over as an already-scaled H6b with the scales
      // set to 1 and the floor set to zero -- see the params impl_aa.py writes.
      in.H6b = dlio::degeneracy::from_upper21(r.lam21);
      in.h_valid = true;
    } else {
      in.H6b = dlio::degeneracy::from_upper21(r.H6b);
    }
    const std::size_t lo = (std::size_t)r.imu_lo, hi = (std::size_t)r.imu_hi;
    in.imu.reserve(hi > lo ? hi - lo : 0);
    for (std::size_t j = lo; j < hi && j < tape.imu.size(); ++j) {
      dlio::smoother::ImuSample s;
      s.stamp = tape.imu[j][0];
      s.accel << tape.imu[j][1], tape.imu[j][2], tape.imu[j][3];
      s.gyro << tape.imu[j][4], tape.imu[j][5], tape.imu[j][6];
      in.imu.push_back(s);
    }

    const dlio::smoother::Solution sol = sm.update(in);
    dlio::smoother::WriteBackReport rep;
    {   // the replay writes nothing into a node, but the ledger is the same
      dlio::smoother::WriteTargets t;
      Eigen::Matrix4f Tf = in.T_gicp.cast<float>();
      t.T = &Tf;
      rep = dlio::smoother::write_back(t, sol, P.alpha, P.blend_increment);
    }
    ledger.standing_max_m = P.standing_offset_max_m;
    ledger.note(sol, rep);

    // overwrite every in-window scan's estimate; what survives at the end is
    // the last one before that scan was marginalised out
    sm.readWindow(&wk, &wp, &wv);
    for (std::size_t j = 0; j < wk.size(); ++j) {
      const std::size_t idx = (std::size_t)wk[j];
      if (idx < final_T.size()) {
        final_T[idx] = wp[j];
        final_v[idx] = wv[j];
        final_seen[idx] = 1;
      }
    }
    if (in.dp6_valid) ++n_reg;
    if (sol.floor_binding) ++n_floor;
    if (sol.psd_projected) ++n_psd;
    n_gap += sol.imu_gaps;
    solve_ms.push_back(sol.solve_ms);
    solve_ms_sum += sol.solve_ms;
    solve_ms_max = std::max(solve_ms_max, sol.solve_ms);

    const Eigen::Matrix3d Rm = sol.T.block<3, 3>(0, 0);
    const Eigen::Quaterniond q(Rm);
    fp << "{\"i\":" << i << ",\"t\":" << r.stamp << ",\"valid\":"
       << (sol.valid ? 1 : 0) << ",\"p\":[" << sol.T(0, 3) << "," << sol.T(1, 3)
       << "," << sol.T(2, 3) << "],\"q\":[" << q.w() << "," << q.x() << ","
       << q.y() << "," << q.z() << "],\"v\":[" << sol.v(0) << "," << sol.v(1)
       << "," << sol.v(2) << "]}\n";
    fs << "{\"i\":" << i << ",\"solve_ms\":" << sol.solve_ms << ",\"win\":"
       << sol.window_vars << ",\"nfac\":" << sol.window_factors << ",\"lm\":"
       << sol.lm_iterations << ",\"imu\":" << sol.imu_samples << ",\"gaps\":"
       << sol.imu_gaps << ",\"corr_m\":" << rep.corr_m << ",\"corr_deg\":"
       << rep.corr_deg << ",\"resid_imu\":" << sol.resid_imu << ",\"resid_reg\":"
       << sol.resid_reg << ",\"mcov\":[" << sol.marg_cov[0] << "," << sol.marg_cov[1]
       << "," << sol.marg_cov[2] << "," << sol.marg_cov[3] << ","
       << sol.marg_cov[4] << "," << sol.marg_cov[5] << "],\"floor\":"
       << (sol.floor_binding ? 1 : 0) << ",\"psd\":" << (sol.psd_projected ? 1 : 0)
       << ",\"exc\":" << (sol.update_exception ? 1 : 0)
       << ",\"tgt_m\":" << rep.target_m << ",\"tgt_deg\":" << rep.target_deg
       << ",\"anch_k\":" << sol.anchor_key << ",\"anch_m\":" << sol.anchor_resid_m
       << ",\"ncorr\":" << r.ncorr << "}\n";
  }
  fp.close();
  fs.close();

  // the FINAL (lag-L) trajectory, beside the lag-0 one
  {
    std::string fin = std::string(argv[3]);
    const std::size_t dot = fin.rfind(".ndjson");
    fin = (dot == std::string::npos) ? (fin + ".final") : (fin.substr(0, dot) + "_final.ndjson");
    std::ofstream ff(fin);
    ff.precision(17);
    long unseen = 0;
    for (std::size_t i = 0; i < final_T.size(); ++i) {
      if (!final_seen[i]) ++unseen;
      const Eigen::Quaterniond q(Eigen::Matrix3d(final_T[i].block<3, 3>(0, 0)));
      ff << "{\"i\":" << i << ",\"t\":" << tape.rows[i].stamp << ",\"valid\":"
         << (int)final_seen[i] << ",\"p\":[" << final_T[i](0, 3) << ","
         << final_T[i](1, 3) << "," << final_T[i](2, 3) << "],\"q\":[" << q.w()
         << "," << q.x() << "," << q.y() << "," << q.z() << "],\"v\":["
         << final_v[i](0) << "," << final_v[i](1) << "," << final_v[i](2) << "]}\n";
    }
    ff.close();
    // [[silent_no_op_law]]: a scan that never got an estimate is a hole, and a
    // hole that is not counted is a hole nobody sees.
    std::fprintf(stderr, "[REPLAY] final trajectory -> %s (unwritten %ld)\n",
                 fin.c_str(), unseen);
  }

  std::sort(solve_ms.begin(), solve_ms.end());
  auto pct = [&](double q) {
    if (solve_ms.empty()) return 0.0;
    std::size_t i = (std::size_t)(q * (solve_ms.size() - 1));
    return solve_ms[i];
  };
  const std::string v = ledger.violation(P.alpha);
  std::fprintf(stderr,
               "[REPLAY] %s | scans %ld solved %ld computed %ld applied %ld "
               "exceptions %ld reseats %ld | reg_factors %ld floor_binding %ld psd_proj %ld "
               "imu_gaps %ld | solve_ms p50 %.3f p95 %.3f max %.3f mean %.3f | %s\n",
               sm.versionString().c_str(), ledger.scans, ledger.solved,
               ledger.computed, ledger.applied, ledger.exceptions, ledger.reseats,
               n_reg, n_floor,
               n_psd, n_gap, pct(0.50), pct(0.95), solve_ms_max,
               solve_ms_sum / std::max<std::size_t>(1, solve_ms.size()),
               v.empty() ? "OK" : ("SILENT-NO-OP: " + v).c_str());
  return v.empty() ? 0 : 1;
}
