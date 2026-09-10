/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/odom.h"

// Local point type mirroring the ouster_ros PointXYZIRT layout, used only when
// reading incoming Ouster clouds so we can grab reflectivity and ring without
// adding them to dlio::Point's hot-path representation.
namespace dlio_ouster {
  struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    std::uint32_t t;
    std::uint16_t reflectivity;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(dlio_ouster::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t)
                                  (std::uint16_t, reflectivity, reflectivity)
                                  (std::uint16_t, ring, ring))

dlio::OdomNode::OdomNode(ros::NodeHandle node_handle) : nh(node_handle) {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->dlio_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;

  // --- E4 / ARCHITECT A -----------------------------------------------------
  this->smoother_bias_seeded_ = false;
  this->smoother_noop_reported_ = false;
  this->smoother_scans_ = 0;
  this->smoother_solve_ms_max_ = 0.;
  this->smoother_kf_applied_ = 0;
  this->smoother_kf_refused_ = 0;
  this->smoother_submap_dirty_ = false;
  this->raw_imu_n_ = 0;
  this->raw_imu_keep_s_ = this->smoother_params_.lag_s + 5.0;
  this->kf_sink_.reset(new dlio::OdomNode::KfSink(this));
  if (this->smoother_params_.enabled) {
    this->smoother_.reset(new dlio::smoother::Smoother(this->smoother_params_));
    // The extrinsic comes from the GENERATED yaml, which getParams() has
    // already read into extrinsics.baselink2imu -- never from the bundle
    // sidecar, which still ships the house7 MicroStrain lever arm
    // ([[rig2860_sidecar_imu_extrinsic_wrong]]).
    this->smoother_->setExtrinsic(this->extrinsics.baselink2imu.R.cast<double>(),
                                  this->extrinsics.baselink2imu.t.cast<double>());
    printf("[SMOOTH] ARMED  %s  lag_s=%.2f alpha=%.3f "
           "info_scale_rot=%.9g info_scale_trans=%.9g huber_k=%.3f "
           "floor=(%.3g deg, %.3g m) marginalize_every=%d kf_writeback=%d\n",
           this->smoother_->versionString().c_str(),
           this->smoother_params_.lag_s, this->smoother_params_.alpha,
           this->smoother_params_.info_scale_rot,
           this->smoother_params_.info_scale_trans, this->smoother_params_.huber_k,
           this->smoother_params_.floor_sigma_rot_deg,
           this->smoother_params_.floor_sigma_trans_m,
           this->smoother_params_.marginalize_every,
           (int)this->smoother_params_.keyframe_writeback);
    // The raw tap feeds gtsam the sample as it arrived; imu_accel_sm_ is DLIO's
    // accelerometer scale-misalignment matrix and it is applied to the buffered
    // copy only. Identity is the shipped value and the only one this tap is
    // correct for, so a non-identity one must stop the run rather than be
    // silently ignored.
    if (!this->imu_accel_sm_.isIdentity(1e-9)) {
      fprintf(stderr,
              "[SMOOTH][ERROR] dlio/imu/intrinsics/accel/sm is not the identity "
              "and the smoother's RAW IMU tap does not apply it. Either ship the "
              "identity or teach the tap. Refusing to run with a silently "
              "different IMU on the two sides.\n");
      fflush(stderr);
      ros::shutdown();
    }
    // Two actions on one scan are unattributable. The guard's ACTION is
    // permanently off in this design (A_nudge sec 3.8): the eigen-summaries
    // stay as telemetry, and the arbitration is the solve. `observe` is not
    // only allowed but REQUIRED -- the smoother's information comes from the
    // E2 record scoreDegeneracy() builds.
    if (this->degen_params_.enabled) {
      fprintf(stderr,
              "[SMOOTH][ERROR] the degeneracy GUARD (dlio/odom/gicp/degeneracy/"
              "enabled) and the smoother are both armed. Four hand-fitted bands "
              "have failed in BOTH directions and the smoother exists to replace "
              "them, not to run underneath one; two actions on one scan are "
              "unattributable. Run the guard in OBSERVE mode.\n");
      fflush(stderr);
      ros::shutdown();
    }
    if (!this->degen_params_.scoring()) {
      fprintf(stderr,
              "[SMOOTH][ERROR] the smoother is armed but the degeneracy record "
              "is OFF (neither enabled nor observe). The registration "
              "information the smoother weighs -- H6b, dp6 -- is built in "
              "scoreDegeneracy(), so with the record off every scan would reach "
              "the graph carrying nothing but the constant floor. Set "
              "dlio/odom/gicp/degeneracy/observe: true.\n");
      fflush(stderr);
      ros::shutdown();
    }
    fflush(stdout);
  }

  // Lidar subscriber on a DEDICATED single-threaded callback queue (see odom.h):
  // a deep queue buffers the scan backlog while its lone spinner thread runs
  // callbackPointCloud strictly serially -> DLIO flushes every scan at its own
  // pace with no drops and no concurrent-callback corruption, regardless of how
  // fast the bag is played.
  ros::SubscribeOptions lidar_opts = ros::SubscribeOptions::create<sensor_msgs::PointCloud2>(
      "pointcloud", this->sub_pointcloud_queue_,
      [this](const sensor_msgs::PointCloud2ConstPtr& m) { this->callbackPointCloud(m); },
      ros::VoidPtr(), &this->lidar_cb_queue);
  lidar_opts.transport_hints = ros::TransportHints().tcpNoDelay();
  this->lidar_sub = this->nh.subscribe(lidar_opts);
  this->lidar_spinner = std::make_shared<ros::AsyncSpinner>(1, &this->lidar_cb_queue);
  this->lidar_spinner->start();

  // IMU stays on the node's global callback queue (multi-threaded spinner), so it
  // keeps draining while the lidar thread is busy on a scan.
  this->imu_sub = this->nh.subscribe("imu", this->sub_imu_queue_,
      &dlio::OdomNode::callbackImu, this, ros::TransportHints().tcpNoDelay());

  // Deep publisher queue: highrate_odom emits a ~10-sample burst per scan (not 1
  // msg/scan), so a queue of 1 silently drops most of a burst before the
  // subscriber's TCP can drain it. Size it to buffer a full bag's worth of 100Hz
  // odom even if the recorder stalls during heavy PCD/image writes (msgs are tiny).
  this->odom_pub     = this->nh.advertise<nav_msgs::Odometry>("odom", 200000, true);
  this->pose_pub     = this->nh.advertise<geometry_msgs::PoseStamped>("pose", 1, true);
  this->path_pub     = this->nh.advertise<nav_msgs::Path>("path", 1, true);
  this->kf_pose_pub  = this->nh.advertise<geometry_msgs::PoseArray>("kf_pose", 1, true);
  this->kf_cloud_pub = this->nh.advertise<sensor_msgs::PointCloud2>("kf_cloud", 1, true);
  this->deskewed_pub = this->nh.advertise<sensor_msgs::PointCloud2>("deskewed", 1, true);

  // Odom/pose are published PER-SCAN from callbackPointCloud (stamped with the
  // scan time), not on a 100Hz wall-clock timer stamped imu_stamp. The timer
  // decoupled the odom stamp from the scan it reflects, so under processing
  // backlog the recorded trajectory got mis-stamped by up to tens of seconds.
  // this->publish_timer = this->nh.createTimer(ros::Duration(0.01), &dlio::OdomNode::publishPose, this);

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->deskewed_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->current_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->submap_cloud = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());

  this->num_processed_keyframes = 0;

  this->submap_hasChanged = true;
  this->submap_kf_idx_prev.clear();

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed;

  // Delivered-IMU accounting (instrument; nothing reads it but the log)
  this->imu_rx_ = dlio::imu_delivery::Counter();
  this->imu_rx_.gap_s = this->imu_gap_s_;
  this->imu_gap_reported_ = false;
  this->imu_rx_n_ = 0;
  this->imu_rx_hz_ = 0.;
  this->imu_rx_maxdt_ = 0.;
  this->imu_rx_gaps_ = 0;

  // GICP degeneracy guard telemetry
  this->degen_flag_ = false;
  this->degen_ratio_min_ = 1.;
  this->degen_w_min_ = 1.;
  this->degen_removed_ = 0.;
  this->degen_removed_total_ = 0.;
  this->degen_innov_ = 0.;
  this->degen_dp_ = 0.;
  this->degen_scans_ = 0;
  this->degen_degenerate_ = 0;
  this->degen_applied_ = 0;
  this->degen_invalid_ = 0;
  this->degen_noop_reported_ = false;
  this->degen_blind_reported_ = false;

  // E2 -- the true 6x6 record. Instrumentation only; every field starts empty
  // and a run that never scores prints nothing new.
  this->degen_h_ = dlio::degeneracy::HessianRecord();
  this->degen_T_prev_.setIdentity();
  this->degen_T_prev_valid_ = false;
  this->degen_dp6_valid_ = false;
  this->degen_dp6_.setZero();
  this->degen_corr6_.setZero();
  this->degen_innov6_.setZero();
  this->degen_h6_written_ = 0;
  this->degen_h6_noop_reported_ = false;

  // INCREMENT 1 repairs (dlio/repairs.h). Every counter starts at zero and the
  // clamp starts at the CONFIGURED constant, so a run that arms nothing is the
  // stock run with three more numbers printed.
  this->submap_short_refused_ = 0;
  this->submap_kcc_added_ = 0;
  this->submap_size_ = 0;
  this->submap_dmax_ = 0.;
  this->submap_dkcc_ = 0.;
  this->submap_refuse_reported_ = false;
  this->keyframe_age_fired_ = 0;
  this->keyframe_age_stale_ = 0;
  this->keyframe_age_last_s_ = 0.;
  this->keyframe_age_noop_reported_ = false;
  this->geo_abias_derived_ = false;
  this->geo_abias_clamp_.setConstant((float)this->geo_abias_max_);

  this->convex_hull.setDimension(3);
  this->concave_hull.setDimension(3);
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);
  this->concave_hull.setKeepInformation(true);

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  this->gicp_temp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp_temp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp_temp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp_temp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp_temp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp_temp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);
  this->gicp_temp.setSearchMethodSource(temp, true);
  this->gicp_temp.setSearchMethodTarget(temp, true);

  // Determinism: single-threaded GICP fixes its OpenMP reduction order, cutting
  // run-to-run FP pose noise ~7x (~11mm -> ~1.5mm; not bit-exact). Opt-in (off =
  // multi-threaded, faster).
  const int gicp_threads = this->deterministic_ ? 1 : this->num_threads_;
  this->gicp.setNumThreads(gicp_threads);
  this->gicp_temp.setNumThreads(gicp_threads);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  this->metrics.spaciousness.push_back(0.);
  this->metrics.density.push_back(this->gicp_max_corr_dist_);

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

}

dlio::OdomNode::~OdomNode() {
  // --- E4: the compute budget, reported rather than assumed. The design's own
  // requirement is that the solve stays under the scan period (100 ms at 10 Hz)
  // for a 5 s window; if it does not, the levers are marginalize_every and the
  // lag, and this line is what says which.
  if (this->smoother_ && !this->smoother_solve_ms_.empty()) {
    std::vector<double> v = this->smoother_solve_ms_;
    std::sort(v.begin(), v.end());
    const auto q = [&v](double f) {
      return v[(std::size_t)(f * (double)(v.size() - 1))];
    };
    double sum = 0.;
    for (double x : v) sum += x;
    const std::string viol =
        this->smoother_ledger_.violation(this->smoother_params_.alpha);
    printf("[SMOOTH] SUMMARY scans=%ld solved=%ld computed=%ld applied=%ld "
           "exceptions=%ld reseats=%ld kf_applied=%ld kf_refused=%ld raw_imu=%ld "
           "solve_ms p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f "
           "budget_100ms_exceeded=%ld verdict=%s "
           "units=solve_ms:ms;budget:scans_whose_solve_exceeded_the_100_ms_"
           "scan_period\n",
           this->smoother_ledger_.scans, this->smoother_ledger_.solved,
           this->smoother_ledger_.computed, this->smoother_ledger_.applied,
           this->smoother_ledger_.exceptions, this->smoother_ledger_.reseats,
           (long)this->smoother_kf_applied_,
           (long)this->smoother_kf_refused_, (long)this->raw_imu_n_,
           q(0.50), q(0.95), q(0.99), v.back(), sum / (double)v.size(),
           (long)std::count_if(v.begin(), v.end(),
                               [](double x) { return x > 100.0; }),
           viol.empty() ? "OK" : "SILENT-NO-OP");
    fflush(stdout);
  }
}

void dlio::OdomNode::getParams() {

  // Version
  ros::param::param<std::string>("~dlio/version", this->version_, "0.0.0");

  // Frames
  ros::param::param<std::string>("~dlio/frames/odom", this->odom_frame, "odom");
  ros::param::param<std::string>("~dlio/frames/baselink", this->baselink_frame, "base_link");
  ros::param::param<std::string>("~dlio/frames/lidar", this->lidar_frame, "lidar");
  ros::param::param<std::string>("~dlio/frames/imu", this->imu_frame, "imu");

  // Get Node NS and Remove Leading Character
  std::string ns = ros::this_node::getNamespace();
  ns.erase(0,1);

  // Concatenate Frame Name Strings
  this->odom_frame = ns + "/" + this->odom_frame;
  this->baselink_frame = ns + "/" + this->baselink_frame;
  this->lidar_frame = ns + "/" + this->lidar_frame;
  this->imu_frame = ns + "/" + this->imu_frame;

  // Deskew FLag
  ros::param::param<bool>("~dlio/pointcloud/deskew", this->deskew_, true);

  // Gravity
  ros::param::param<double>("~dlio/odom/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  ros::param::param<bool>("~dlio/odom/computeTimeOffset", this->time_offset_, false);

  // Keyframe Threshold
  ros::param::param<double>("~dlio/odom/keyframe/threshD", this->keyframe_thresh_dist_, 0.1);
  ros::param::param<double>("~dlio/odom/keyframe/threshR", this->keyframe_thresh_rot_, 1.0);

  // Submap
  ros::param::param<int>("~dlio/odom/submap/keyframe/knn", this->submap_knn_, 10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcv", this->submap_kcv_, 10);
  ros::param::param<int>("~dlio/odom/submap/keyframe/kcc", this->submap_kcc_, 10);

  // INCREMENT 1b -- THE KEYFRAME AGE CLAUSE. DEFAULT 0 = OFF, so a row that
  // sets neither key produces a byte-identical parameter dump and a
  // byte-identical keyframe decision to stock DLIO. See dlio/repairs.h and
  // keeper/sandland_2860/big_bag/submap_check/SUBMAP.md section 5: on a revisit
  // the closest keyframe of ANY age is 0.18-0.81 m away and 139-158 s old, the
  // rotation escape hatch is closed by num_nearby >= 3, and the run makes NO
  // keyframe for 28.80 s while believing it walked 15.77 m and turned 162 deg.
  ros::param::param<double>("~dlio/odom/keyframe/max_age_s",
                            this->keyframe_age_.max_age_s, 0.0);
  // Negative = derive 0.25 * threshD. Read AFTER threshD on purpose.
  ros::param::param<double>("~dlio/odom/keyframe/min_travel_m",
                            this->keyframe_age_.min_travel_m, -1.0);
  this->keyframe_age_travel_m_ =
      this->keyframe_age_.travel_floor(this->keyframe_thresh_dist_);

  // Dense map resolution
  ros::param::param<bool>("~dlio/map/dense/filtered", this->densemap_filtered_, true);

  // Full-density output: publish/save the NON-voxelized deskewed scan (default on).
  // Overrides densemap_filtered_ for the published `deskewed` topic only; GICP,
  // keyframes and the map are unaffected (they use the voxelized current_scan).
  ros::param::param<bool>("~dlio/pointcloud/dense_output", this->dense_output_, true);

  // Wait until movement to publish map
  ros::param::param<bool>("~dlio/map/waitUntilMove", this->wait_until_move_, false);

  // Crop Box Filter
  ros::param::param<double>("~dlio/odom/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  ros::param::param<bool>("~dlio/pointcloud/voxelize", this->vf_use_, true);
  ros::param::param<double>("~dlio/odom/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Adaptive Parameters
  ros::param::param<bool>("~dlio/adaptive", this->adaptive_params_, true);

  // Force single-threaded GICP for bit-identical, reproducible poses (slower).
  ros::param::param<bool>("~dlio/deterministic", this->deterministic_, false);

  // High-rate (data-driven) odom: dense IMU-rate poses on the odom topic.
  ros::param::param<bool>("~dlio/highrate_odom", this->publish_highrate_odom_, false);
  double highrate_hz;
  ros::param::param<double>("~dlio/highrate_odom_hz", highrate_hz, 100.0);
  this->highrate_odom_dt_ = (highrate_hz > 0.0) ? (1.0 / highrate_hz) : 0.01;
  this->highrate_anchor_valid_ = false;
  this->highrate_prev_stamp_ = 0.0;
  this->highrate_prev_p_ = Eigen::Vector3f(0., 0., 0.);
  this->highrate_prev_q_ = Eigen::Quaternionf(1., 0., 0., 0.);
  this->highrate_prev_v_ = Eigen::Vector3f(0., 0., 0.);

  // Extrinsics
  std::vector<float> t_default{0., 0., 0.};
  std::vector<float> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<float> baselink2imu_t, baselink2imu_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2imu_R.data(), 3, 3);

  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<float> baselink2lidar_t, baselink2lidar_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2lidar_R.data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  ros::param::param<bool>("~dlio/odom/imu/calibration/accel", this->calibrate_accel_, true);
  ros::param::param<bool>("~dlio/odom/imu/calibration/gyro", this->calibrate_gyro_, true);
  ros::param::param<double>("~dlio/odom/imu/calibration/time", this->imu_calib_time_, 3.0);
  ros::param::param<int>("~dlio/odom/imu/bufferSize", this->imu_buffer_size_, 2000);

  // Subscriber queue depths (see odom.h). Defaults preserve live behavior; the
  // recorder launch sets these large for drop-free deterministic bag processing.
  ros::param::param<int>("~dlio/sub_pointcloud_queue", this->sub_pointcloud_queue_, 1);
  ros::param::param<int>("~dlio/sub_imu_queue", this->sub_imu_queue_, 1000);
  // What counts as a GAP in the delivered IMU stream. 50 ms is the figure the
  // big-bag IMU census counted 2,048 of (IMU_CHECK.md section 5), so the node's
  // own report and that census are in the same unit and can be compared line
  // for line. It changes no arithmetic: it only sets when the counter says so.
  ros::param::param<double>("~dlio/odom/imu/gap_report_s", this->imu_gap_s_, 0.05);

  // --- E4 / ARCHITECT A: the fixed-lag smoother ------------------------------
  // ENABLED IS FALSE IN THE IMAGE. The recipe row turns it on, so the product
  // path is untouched by the presence of this code, and a row that forgets the
  // key gets stock DLIO rather than an unannounced estimator change.
  // Every key below is proved APPLIED from the LIVE rosparam dump per draw
  // (E4/IMPL/rosparam_proof.sh), never from the row (N58).
  {
    dlio::smoother::Params& S = this->smoother_params_;
    ros::param::param<bool>  ("~dlio/smoother/enabled", S.enabled, false);
    ros::param::param<double>("~dlio/smoother/lag_s", S.lag_s, 5.0);
    ros::param::param<double>("~dlio/smoother/alpha", S.alpha, 0.5);
    ros::param::param<double>("~dlio/smoother/info_scale_rot", S.info_scale_rot,
                              0.031503901758232755);
    ros::param::param<double>("~dlio/smoother/info_scale_trans", S.info_scale_trans,
                              0.031503901758232755);
    ros::param::param<double>("~dlio/smoother/huber_k", S.huber_k, 1.345);
    ros::param::param<double>("~dlio/smoother/floor_sigma_trans_m",
                              S.floor_sigma_trans_m, 10.0);
    ros::param::param<double>("~dlio/smoother/floor_sigma_rot_deg",
                              S.floor_sigma_rot_deg, 30.0);
    ros::param::param<int>   ("~dlio/smoother/marginalize_every",
                              S.marginalize_every, 1);
    ros::param::param<int>   ("~dlio/smoother/marg_every", S.marg_every, 1);
    ros::param::param<bool>  ("~dlio/smoother/keyframe_writeback",
                              S.keyframe_writeback, true);
    ros::param::param<double>("~dlio/smoother/kf_dirty_trans_m",
                              S.kf_dirty_trans_m, 0.005);
    ros::param::param<double>("~dlio/smoother/kf_dirty_rot_deg",
                              S.kf_dirty_rot_deg, 0.05);
    ros::param::param<int>   ("~dlio/smoother/lm_max_iterations",
                              S.lm_max_iterations, 20);
    ros::param::param<double>("~dlio/smoother/lm_relative_error_tol",
                              S.lm_relative_error_tol, 1e-8);
    ros::param::param<std::string>("~dlio/smoother/linear_solver",
                                   S.linear_solver, std::string("MULTIFRONTAL_CHOLESKY"));
    // A sample interval longer than this is DECLARED and counted rather than
    // hidden. 3 ticks at 640 Hz = 4.7 ms.
    ros::param::param<double>("~dlio/smoother/imu_gap_s", S.imu_gap_s,
                              3.0 / 640.0);
    ros::param::param<int>   ("~dlio/smoother/log_every", S.log_every, 1);
    ros::param::param<double>("~dlio/smoother/gauge_sigma_rp_deg",
                              S.gauge_sigma_rp_deg, 0.5);
    ros::param::param<double>("~dlio/smoother/gauge_sigma_yaw_deg",
                              S.gauge_sigma_yaw_deg, 10.0);
    ros::param::param<double>("~dlio/smoother/gauge_sigma_pos_m",
                              S.gauge_sigma_pos_m, 10.0);
    ros::param::param<double>("~dlio/smoother/v0_sigma", S.v0_sigma, 0.5);
    ros::param::param<double>("~dlio/smoother/bias_prior_sigma_accel",
                              S.bias_prior_sigma_accel, 0.05);
    ros::param::param<double>("~dlio/smoother/bias_prior_sigma_gyro",
                              S.bias_prior_sigma_gyro, 0.005);
    // The rig's OWN measured Allan analysis, per axis, sensor frame. Exposed so
    // a DIFFERENT rig can be given its own measurement -- never so this one can
    // be tuned. reeval/check3_noise.json is the authority for rig 2860.
    std::vector<double> vrw_d(S.vrw, S.vrw + 3), arw_d(S.arw, S.arw + 3);
    std::vector<double> aad(S.acc_adev_2s, S.acc_adev_2s + 3);
    std::vector<double> gad(S.gyr_adev_2s, S.gyr_adev_2s + 3);
    std::vector<double> vrw_o, arw_o, aa_o, ga_o;
    ros::param::param<std::vector<double>>("~dlio/smoother/imu/vrw", vrw_o, vrw_d);
    ros::param::param<std::vector<double>>("~dlio/smoother/imu/arw", arw_o, arw_d);
    ros::param::param<std::vector<double>>("~dlio/smoother/imu/acc_adev_2s", aa_o, aad);
    ros::param::param<std::vector<double>>("~dlio/smoother/imu/gyr_adev_2s", ga_o, gad);
    for (int i = 0; i < 3; ++i) {
      if (vrw_o.size() == 3) S.vrw[i] = vrw_o[i];
      if (arw_o.size() == 3) S.arw[i] = arw_o[i];
      if (aa_o.size() == 3) S.acc_adev_2s[i] = aa_o[i];
      if (ga_o.size() == 3) S.gyr_adev_2s[i] = ga_o[i];
    }
    S.gravity = this->gravity_;
    if (S.alpha < 0.0 || S.alpha > 1.0) {
      fprintf(stderr, "[SMOOTH][ERROR] alpha=%.4f is outside [0,1]. It is a "
                      "BLEND, not a gain, and it is never fitted.\n", S.alpha);
      fflush(stderr);
      S.alpha = std::min(1.0, std::max(0.0, S.alpha));
    }
  }

  std::vector<float> accel_default{0., 0., 0.}; std::vector<float> prior_accel_bias;
  std::vector<float> gyro_default{0., 0., 0.}; std::vector<float> prior_gyro_bias;

  ros::param::param<bool>("~dlio/odom/imu/approximateGravity", this->gravity_align_, true);
  ros::param::param<bool>("~dlio/imu/calibration", this->imu_calibrate_, true);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<float> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<float> imu_sm;

  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(imu_sm.data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  ros::param::param<int>("~dlio/odom/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  ros::param::param<int>("~dlio/odom/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  ros::param::param<double>("~dlio/odom/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  ros::param::param<int>("~dlio/odom/gicp/maxIterations", this->gicp_max_iter_, 64);
  ros::param::param<double>("~dlio/odom/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  ros::param::param<double>("~dlio/odom/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  ros::param::param<double>("~dlio/odom/geo/Kp", this->geo_Kp_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kv", this->geo_Kv_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kq", this->geo_Kq_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kab", this->geo_Kab_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/Kgb", this->geo_Kgb_, 1.0);
  ros::param::param<double>("~dlio/odom/geo/abias_max", this->geo_abias_max_, 1.0);
  // N57 -- THE CLAMP MUST NOT BE AN ABSOLUTE CONSTANT. 0.3 is BELOW this rig's
  // own turn-on bias (Octagon init 0.305; big bag 0.236-0.251), so it clips the
  // CALIBRATED value. With a positive margin the bound becomes |b_init| + margin
  // PER AXIS, measured by this run's own 3 s init calibration. 0 = off = the
  // constant above, unchanged, which is what every existing row gets.
  ros::param::param<double>("~dlio/odom/geo/abias_margin", this->geo_abias_margin_, 0.0);
  // The derivation has exactly one source. If accel calibration is off there is
  // no |b_init| to derive from and the margin would silently do nothing.
  if (this->geo_abias_margin_ > 0.0 && !this->calibrate_accel_) {
    fprintf(stderr,
            "[REPAIR][ERROR] dlio/odom/geo/abias_margin=%.4f asks for a clamp "
            "derived from the init calibration, but dlio/odom/imu/calibration/"
            "accel is FALSE, so no bias is ever measured. The clamp would stay "
            "at the constant abias_max=%.4f and the margin would be a silent "
            "no-op. Set one or the other.\n",
            this->geo_abias_margin_, this->geo_abias_max_);
    fflush(stderr);
  }
  ros::param::param<double>("~dlio/odom/geo/gbias_max", this->geo_gbias_max_, 1.0);

  // GICP degeneracy guard. EVERY default here reproduces stock DLIO: with
  // enabled=false and observe=false nothing is scored, nothing is logged, and
  // updateState()/getNextPose() run exactly the arithmetic they always did.
  ros::param::param<bool>  ("~dlio/odom/gicp/degeneracy/enabled",       this->degen_params_.enabled,       false);
  ros::param::param<bool>  ("~dlio/odom/gicp/degeneracy/observe",       this->degen_params_.observe,       false);
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/min_ratio",     this->degen_params_.min_ratio,     0.02);
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/full_ratio",    this->degen_params_.full_ratio,    0.10);
  // ABSOLUTE per-point information band, q = lambda_k/ncorr. 0/0 = OFF, which is
  // what every row that predates this key gets: the ratio test alone, unchanged.
  // See dlio/degeneracy.h for why the ratio needs a companion at all.
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/min_info",      this->degen_params_.min_info,      0.0);
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/full_info",     this->degen_params_.full_info,     0.0);
  ros::param::param<int>   ("~dlio/odom/gicp/degeneracy/max_weak_dirs", this->degen_params_.max_weak_dirs, 1);
  // The weight FLOOR. Defaults to 0.25, not 0: w = 0 hands the weak axis to a
  // free double integrator for the whole degenerate stretch (780 s on the
  // Sandland corridor) with no substitute constraint anywhere in DLIO, which is
  // hundreds of metres of drift, not the bounded linear drift this guard claims.
  // See dlio/degeneracy.h. 0.0 = literal solution remapping, opt in deliberately.
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/min_weight",    this->degen_params_.min_weight,    0.25);
  ros::param::param<int>   ("~dlio/odom/gicp/degeneracy/min_corr",      this->degen_params_.min_corr,      200);
  ros::param::param<bool>  ("~dlio/odom/gicp/degeneracy/guard_pose",    this->degen_params_.guard_pose,    false);
  ros::param::param<double>("~dlio/odom/gicp/degeneracy/innov_max_m",   this->degen_params_.innov_max_m,   0.0);
  ros::param::param<int>   ("~dlio/odom/gicp/degeneracy/log_every",     this->degen_params_.log_every,     1);

  ros::param::param<bool>("~dlio/verbose", this->verbose, true);
}

void dlio::OdomNode::start() {
  if (!this->verbose) {
    return;
  }

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

// High-rate (~100Hz) odom for the JUST-FINISHED scan interval. This replaces the
// removed wall-clock 100Hz publish_timer: it is DATA-DRIVEN (called once per
// processed scan from callbackPointCloud, so it flushes at the processing rate
// and never lag-warps), BOUNDED (only the one scan interval's worth of samples,
// so a backlog can't make it run away), and READ-ONLY (it integrates a private
// copy of the IMU prior off the previous corrected anchor and never touches
// this->state / the geo-observer, so it can't cause the divergence the old
// real-time propagate did). It emits dense poses on the SAME odom topic, stamped
// at uniform-decimated times strictly INSIDE (prev_corrected_stamp, scan_stamp);
// the corrected per-scan pose at exactly scan_stamp is published by publishPose()
// right after, so the exporter's nearest-stamp pairing still snaps clouds to the
// corrected pose (dt=0 wins) while odom.json gets the full ~100Hz stream for
// B-spline interpolation. The anchor is re-set to the corrected this->state at
// the end of publishPose(), so each interval is integrated fresh off a corrected
// pose (no unbounded drift; a tiny discontinuity at each scan boundary is fine
// for interpolation).
void dlio::OdomNode::publishHighRateInterval() {
  if (!this->publish_highrate_odom_ || !this->highrate_anchor_valid_) return;
  if (this->scan_stamp <= this->highrate_prev_stamp_) return;

  // Uniform decimated sample times strictly inside the interval. Stop HALF a dt
  // before scan_stamp so the final high-rate sample can't land within a few us of
  // it (which would make a near-zero-dt pair in odom.json that differs by the
  // geo-observer correction jump -- a spurious spike for B-spline fitting). The
  // corrected pose at exactly scan_stamp, emitted by publishPose() right after,
  // owns that final point of the interval.
  const double t_end = this->scan_stamp - 0.5 * this->highrate_odom_dt_;
  std::vector<double> ts;
  for (double t = this->highrate_prev_stamp_ + this->highrate_odom_dt_;
       t < t_end; t += this->highrate_odom_dt_) {
    ts.push_back(t);
  }
  if (ts.empty()) return;

  // Forward-integrate the IMU prior from the previous corrected anchor. Returns
  // world poses at each ts; empty if the IMU window isn't available (lag/gap) ->
  // we simply skip this interval rather than emit anything wrong.
  auto frames = this->integrateImu(this->highrate_prev_stamp_, this->highrate_prev_q_,
                                   this->highrate_prev_p_, this->highrate_prev_v_, ts);
  if (frames.size() != ts.size()) return;

  nav_msgs::Odometry o;
  o.header.frame_id = this->odom_frame;
  o.child_frame_id = this->baselink_frame;
  for (size_t i = 0; i < ts.size(); ++i) {
    const Eigen::Matrix4f& T = frames[i];
    o.header.stamp = ros::Time(ts[i]);
    o.pose.pose.position.x = T(0,3);
    o.pose.pose.position.y = T(1,3);
    o.pose.pose.position.z = T(2,3);
    Eigen::Quaternionf q(T.block<3,3>(0,0));
    q.normalize();
    o.pose.pose.orientation.w = q.w();
    o.pose.pose.orientation.x = q.x();
    o.pose.pose.orientation.y = q.y();
    o.pose.pose.orientation.z = q.z();
    this->odom_pub.publish(o);
  }
}

void dlio::OdomNode::publishPose() {

  // Emit the dense ~100Hz poses for the interval that just closed (samples are
  // strictly before scan_stamp), THEN the corrected per-scan pose below.
  this->publishHighRateInterval();

  // nav_msgs::Odometry. Stamped with scan_stamp = the MEDIAN point time, which is
  // the time this->state actually represents (the geo-observer is propagated to
  // scan_stamp and corrected toward lidarPose = T_corr*frames[median]). NOT
  // scan_header_stamp (the scan FRONT) -- that header is ~50ms earlier than the
  // pose value and would mislabel it. NOT imu_stamp -- this is called once per
  // processed scan from callbackPointCloud, matching the deskewed cloud's stamp.
  this->odom_ros.header.stamp = ros::Time(this->scan_stamp);
  this->odom_ros.header.frame_id = this->odom_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub.publish(this->odom_ros);

  // geometry_msgs::PoseStamped (stamped at scan_stamp = MEDIAN = the time the pose
  // value represents; see the odom stamp above).
  this->pose_ros.header.stamp = ros::Time(this->scan_stamp);
  this->pose_ros.header.frame_id = this->odom_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub.publish(this->pose_ros);

  // Re-anchor the high-rate stream to THIS scan's corrected state at scan_stamp,
  // so the next interval integrates fresh off a corrected pose (read-only copy;
  // does not feed back into the SLAM state). v.lin.w is the world-frame velocity.
  this->highrate_prev_stamp_ = this->scan_stamp;
  this->highrate_prev_p_ = this->state.p;
  this->highrate_prev_q_ = this->state.q;
  this->highrate_prev_v_ = this->state.v.lin.w;
  this->highrate_anchor_valid_ = true;

}

void dlio::OdomNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::Path
  this->path_ros.header.stamp = ros::Time(this->scan_stamp);  // scan time (the pose's time), not imu_stamp -- so TF/path keep advancing with the scans being drained, even after the bag (IMU) stops
  this->path_ros.header.frame_id = this->odom_frame;

  geometry_msgs::PoseStamped p;
  p.header.stamp = ros::Time(this->scan_stamp);  // scan time (the pose's time), not imu_stamp -- so TF/path keep advancing with the scans being drained, even after the bag (IMU) stops
  p.header.frame_id = this->odom_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub.publish(this->path_ros);

  // transform: odom to baselink
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = ros::Time(this->scan_stamp);  // scan time (the pose's time), not imu_stamp -- so TF/path keep advancing with the scans being drained, even after the bag (IMU) stops
  transformStamped.header.frame_id = this->odom_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = ros::Time(this->scan_stamp);  // scan time (the pose's time), not imu_stamp -- so TF/path keep advancing with the scans being drained, even after the bag (IMU) stops
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = ros::Time(this->scan_stamp);  // scan time (the pose's time), not imu_stamp -- so TF/path keep advancing with the scans being drained, even after the bag (IMU) stops
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br.sendTransform(transformStamped);

}

void dlio::OdomNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  if (this->wait_until_move_) {
    if (this->length_traversed < 0.1) { return; }
  }

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ (boost::make_shared<pcl::PointCloud<PointType>>());

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  // Stamp at scan_stamp = MEDIAN point time (the cloud's rigid world anchor is
  // T_corr*frames[median], and it must match the odom/pose stamp that labels the
  // same-time pose value). NOT scan_header_stamp (the scan FRONT, ~50ms earlier).
  deskewed_ros.header.stamp = ros::Time(this->scan_stamp);
  deskewed_ros.header.frame_id = this->odom_frame;
  this->deskewed_pub.publish(deskewed_ros);

}

void dlio::OdomNode::publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, pcl::PointCloud<PointType>::ConstPtr> kf, ros::Time timestamp) {

  // Push back
  geometry_msgs::Pose p;
  p.position.x = kf.first.first[0];
  p.position.y = kf.first.first[1];
  p.position.z = kf.first.first[2];
  p.orientation.w = kf.first.second.w();
  p.orientation.x = kf.first.second.x();
  p.orientation.y = kf.first.second.y();
  p.orientation.z = kf.first.second.z();
  // publishKeyframe runs on a DETACHED thread, one per unprocessed keyframe, so
  // several can be inside this function at once. roscpp serializes in two passes
  // (measure the message, then write it into an exactly-sized buffer); a
  // push_back landing between the passes overruns the buffer and throws
  // StreamOverrunException out of a thread with no handler => std::terminate.
  // That is the abort seen four times on the big Sandland bag, always inside a
  // burst of one-keyframe-per-scan. Mutate under the lock, then serialize a
  // PRIVATE COPY outside it: identical message content, no race, no knob —
  // "please corrupt my message buffer" is not an option anyone should have.
  geometry_msgs::PoseArray kf_msg;
  {
    std::lock_guard<std::mutex> lock(this->kf_pose_mutex);
    this->kf_pose_ros.poses.push_back(p);
    this->kf_pose_ros.header.stamp = timestamp;
    this->kf_pose_ros.header.frame_id = this->odom_frame;
    kf_msg = this->kf_pose_ros;
  }

  // Publish
  this->kf_pose_pub.publish(kf_msg);

  // publish keyframe scan for map
  if (this->vf_use_) {
    if (kf.second->points.size() == kf.second->width * kf.second->height) {
      sensor_msgs::PointCloud2 keyframe_cloud_ros;
      pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
      keyframe_cloud_ros.header.stamp = timestamp;
      keyframe_cloud_ros.header.frame_id = this->odom_frame;
      this->kf_cloud_pub.publish(keyframe_cloud_ros);
    }
  } else {
    sensor_msgs::PointCloud2 keyframe_cloud_ros;
    pcl::toROSMsg(*kf.second, keyframe_cloud_ros);
    keyframe_cloud_ros.header.stamp = timestamp;
    keyframe_cloud_ros.header.frame_id = this->odom_frame;
    this->kf_cloud_pub.publish(keyframe_cloud_ros);
  }

}

void dlio::OdomNode::getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc) {

  // Pre-scan fields so we can route Ouster clouds through a native point type
  // that exposes reflectivity (mapped into intensity) alongside ring.
  bool is_ouster = false;
  for (const auto &field : pc->fields) {
    if (field.name == "t") { is_ouster = true; break; }
  }

  pcl::PointCloud<PointType>::Ptr original_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());

  if (is_ouster) {
    pcl::PointCloud<dlio_ouster::Point> ouster_scan;
    pcl::fromROSMsg(*pc, ouster_scan);

    original_scan_->header = ouster_scan.header;
    original_scan_->height = ouster_scan.height;
    original_scan_->width = ouster_scan.width;
    original_scan_->is_dense = ouster_scan.is_dense;
    original_scan_->points.resize(ouster_scan.points.size());
    for (size_t i = 0; i < ouster_scan.points.size(); ++i) {
      const auto& src = ouster_scan.points[i];
      auto& dst = original_scan_->points[i];
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
      dst.intensity = static_cast<float>(src.reflectivity);
      dst.t = src.t;
      dst.ring = src.ring;
    }
  } else {
    // PCL fills matching named fields (xyz, intensity, time/timestamp, and ring
    // when the source provides it — e.g. Velodyne, Hesai).
    pcl::fromROSMsg(*pc, *original_scan_);
  }

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  this->sensor = dlio::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = dlio::SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = dlio::SensorType::VELODYNE;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp < 1e14) {
      this->sensor = dlio::SensorType::HESAI;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp > 1e14) {
      this->sensor = dlio::SensorType::LIVOX;
      break;
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void dlio::OdomNode::preprocessPoints() {

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = this->scan_header_stamp.toSec();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
      frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_
      (boost::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan));
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::OdomNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_scan_->points.resize(this->original_scan->points.size());

  // individual point timestamps should be relative to this time
  double sweep_ref_time = this->scan_header_stamp.toSec();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };

  } else if (this->sensor == dlio::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    ROS_FATAL("Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::OdomNode::initializeInputTarget() {

  this->prev_scan_stamp = this->scan_stamp;

  // keep history of keyframes
  this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
  this->keyframe_timestamps.push_back(this->scan_header_stamp);
  this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
  this->keyframe_transformations.push_back(this->T_corr);

    // --- E4: register the keyframe with the smoother. A keyframe IS a scan, so
    // there is no new variable: the smoother records which of its own keys this
    // keyframe's pose IS, and reads it back while that key is still inside the
    // window. The parallel vectors below are the map side of the same fact.
    {
      std::lock_guard<std::mutex> lk(this->kf_delta_mutex_);
      this->kf_pending_delta_.push_back(Eigen::Matrix4d::Identity());
      this->kf_has_delta_.push_back(0);
      Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();
      P0.block<3, 3>(0, 0) = this->lidarPose.q.toRotationMatrix().cast<double>();
      P0.block<3, 1>(0, 3) = this->lidarPose.p.cast<double>();
      this->kf_pose_now_.push_back(P0);
      this->kf_frozen_.push_back(0);
    }
    if (this->smoother_) {
      this->smoother_->noteKeyframe((int)this->keyframes.size() - 1);
    }

}

void dlio::OdomNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::OdomNode::initializeDLIO() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  this->dlio_initialized = true;
  std::cout << std::endl << " DLIO initialized!" << std::endl;

}

void dlio::OdomNode::callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc) {

  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(main_loop_running_mutex);
  this->main_loop_running = true;
  lock.unlock();

  double then = ros::Time::now().toSec();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = pc->header.stamp.toSec();
  }

  // DLIO Initialization procedures (IMU calib, gravity align)
  if (!this->dlio_initialized) {
    this->initializeDLIO();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    ROS_FATAL("Low number of points in the cloud!");
    return;
  }

  // Compute Metrics
  this->metrics_thread = std::thread( &dlio::OdomNode::computeMetrics, this );
  this->metrics_thread.detach();

  // Set Adaptive Parameters
  if (this->adaptive_params_) {
    this->setAdaptiveParams();
  }

  // Set new frame as input source
  this->setInputSource();

  // Set initial frame as first keyframe
  if (this->keyframes.size() == 0) {
    this->initializeInputTarget();
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
    this->submap_future.wait(); // wait until completion
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Update current keyframe poses and map
  this->updateKeyframes();

  // Build keyframe normals and submap if needed (and if we're not already waiting)
  if (this->new_submap_is_ready) {
    this->main_loop_running = false;
    this->submap_future =
      std::async( std::launch::async, &dlio::OdomNode::buildKeyframesAndSubmap, this, this->state );
  } else {
    lock.lock();
    this->main_loop_running = false;
    lock.unlock();
    this->submap_build_cv.notify_one();
  }

  // Update trajectory
  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );

  // Update time stamps
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish odom for THIS scan's corrected state, stamped with the scan time,
  // synchronously and BEFORE the detached cloud-publish thread — so a downstream
  // recorder sees "odom then cloud" lockstep per scan regardless of processing lag.
  this->publishPose();

  // Publish stuff to ROS — SYNCHRONOUSLY, in-order, within the callback. The old
  // version spawned a DETACHED thread that read this->scan_header_stamp / state /
  // T_corr as MEMBERS; the next callback would overwrite scan_header_stamp before
  // the detached thread read it, so two distinct deskewed clouds could be
  // published with the SAME stamp (a duplicate scan, playback-speed-dependent;
  // it also raced path_ros). Publishing inline binds the cloud to THIS scan's
  // stamp deterministically. Cost: ~a few ms of toROSMsg/publish added to the
  // ~100 ms scan — negligible, and the recorder is lag-tolerant.
  // Select the cloud to publish/save. dense_output_ (default true) wins: emit the
  // FULL-DENSITY deskewed_scan (motion-compensated, crop+NaN only, no voxel) so the
  // saved PCDs / deskewed topic are the finished dense product. GICP/keyframes/map
  // are NOT affected (they consume the voxelized current_scan). With dense_output_
  // off, fall back to the legacy densemap_filtered_ behavior.
  pcl::PointCloud<PointType>::ConstPtr published_cloud;
  if (this->dense_output_) {
    published_cloud = this->deskewed_scan;
  } else if (this->densemap_filtered_) {
    published_cloud = this->current_scan;
  } else {
    published_cloud = this->deskewed_scan;
  }
  this->publishToROS(published_cloud, this->T_corr);

  // Update some statistics
  this->comp_times.push_back(ros::Time::now().toSec() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements and publish custom DLIO message
  if (this->verbose) {
    this->debug_thread = std::thread( &dlio::OdomNode::debug, this );
    this->debug_thread.detach();
  }
  
  this->geo.first_opt_done = true;
}

void dlio::OdomNode::callbackImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {

  this->first_imu_received = true;

  // --- E4: THE RAW TAP, three lines, BEFORE anything touches the sample -----
  // The sample as it arrived, in the IMU's OWN frame, with no bias removed and
  // no lever-arm term. gtsam's body_P_sensor does the frame change analytically
  // and the CombinedImuFactor subtracts the bias it is currently estimating.
  // DLIO's own imu_buffer is untouched by this and keeps its own arithmetic.
  if (this->smoother_) {
    dlio::OdomNode::RawImu r;
    r.stamp = imu_raw->header.stamp.toSec();
    r.accel << imu_raw->linear_acceleration.x, imu_raw->linear_acceleration.y,
        imu_raw->linear_acceleration.z;
    r.gyro << imu_raw->angular_velocity.x, imu_raw->angular_velocity.y,
        imu_raw->angular_velocity.z;
    std::lock_guard<std::mutex> lk(this->mtx_raw_imu_);
    this->raw_imu_buffer_.push_back(r);
    ++this->raw_imu_n_;
    const double keep = r.stamp - this->raw_imu_keep_s_;
    while (!this->raw_imu_buffer_.empty() &&
           this->raw_imu_buffer_.front().stamp < keep) {
      this->raw_imu_buffer_.pop_front();
    }
  }

  sensor_msgs::Imu::Ptr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;

  // DELIVERED-IMU ACCOUNTING. Before the calibration branch, so the 3 s
  // calibration window is inside the count: the question this answers is how
  // much of the stream REACHED this node, and a sample consumed by calibration
  // reached it. No arithmetic below reads any of this.
  {
    const bool gap = this->imu_rx_.note(imu->header.stamp.toSec());
    this->imu_rx_n_     = this->imu_rx_.n;
    this->imu_rx_hz_    = this->imu_rx_.hz();
    this->imu_rx_maxdt_ = this->imu_rx_.max_dt;
    this->imu_rx_gaps_  = this->imu_rx_.gaps;
    // ONCE, on stderr, the moment the stream proves itself lossy. The banner
    // averages and the [DEGEN] line is sampled; this fires on the first gap and
    // names it, so a lossy transport cannot reach the end of a run unremarked.
    // On a stream that is not losing messages it never prints at all -- which is
    // what makes it evidence.
    if (gap && !this->imu_gap_reported_) {
      this->imu_gap_reported_ = true;
      fprintf(stderr,
              "[IMU][WARN] %.1f ms with no IMU sample, ending at t=%.4f (%ld "
              "received so far, %.2f Hz delivered). The publisher feeding this "
              "node is dropping messages: a mean-of-1/dt rate CANNOT see that, "
              "so judge delivery by the count/span figure in the banner, not by "
              "'Sensor Rates'. Threshold dlio/odom/imu/gap_report_s = %.3f s.\n",
              this->imu_rx_.max_dt * 1e3, this->imu_rx_.max_dt_at,
              (long)this->imu_rx_.n, this->imu_rx_.hz(), this->imu_gap_s_);
      fflush(stderr);
    }
  }

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu->header.stamp.toSec();
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu->header.stamp.toSec() - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // set gravity aligned orientation
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      // N57 -- THE CLAMP, FROM THIS RUN'S OWN CALIBRATION. Derived here and
      // nowhere else, exactly once, from the value the lines above just
      // measured; with abias_margin = 0 (the default) this returns the
      // configured constant on all three axes and updateState()'s arithmetic
      // is bit-for-bit what it always was. See dlio/repairs.h.
      this->geo_abias_clamp_ = dlio::repairs::derive_abias_clamp(
          this->state.b.accel, this->geo_abias_margin_, this->geo_abias_max_);
      this->geo_abias_derived_ = true;

      // --- E4: the smoother's bias prior is THIS RUN'S OWN 3 s init ---------
      // state.b is in BASELINK (transformImu rotated the samples before the
      // averages were taken); the CombinedImuFactor wants it in the SENSOR
      // frame, because that is where it subtracts it from the raw sample. One
      // conversion, in one place, with a fixture (test_smoother case 12).
      if (this->smoother_) {
        const Eigen::Vector3d ba = dlio::smoother::bias_sensor_from_bl(
            this->extrinsics.baselink2imu.R, this->state.b.accel);
        const Eigen::Vector3d bg = dlio::smoother::bias_sensor_from_bl(
            this->extrinsics.baselink2imu.R, this->state.b.gyro);
        this->smoother_->setBiasPrior(ba, bg);
        this->smoother_bias_seeded_ = true;
        std::lock_guard<std::mutex> print_lock(this->print_mutex_);
        printf("[SMOOTH] bias prior from this run's %.1f s init, SENSOR frame: "
               "accel (%.8f, %.8f, %.8f) m/s^2  gyro (%.8f, %.8f, %.8f) rad/s "
               "sigma_a=%.4g sigma_g=%.4g (a seed, not an assertion; there is "
               "NO clamp and NO rail on it anywhere in the graph)\n",
               this->imu_calib_time_, ba(0), ba(1), ba(2), bg(0), bg(1), bg(2),
               this->smoother_params_.bias_prior_sigma_accel,
               this->smoother_params_.bias_prior_sigma_gyro);
        fflush(stdout);
      }
      if (this->geo_abias_margin_ > 0.0) {
        std::lock_guard<std::mutex> print_lock(this->print_mutex_);
        printf("[REPAIR] abias clamp DERIVED from this run's %.1f s init "
               "calibration: |b_init| + margin %.4f -> (%.4f, %.4f, %.4f) "
               "units=m/s^2 (the configured constant %.4f is NOT used)\n",
               this->imu_calib_time_, this->geo_abias_margin_,
               (double)this->geo_abias_clamp_[0], (double)this->geo_abias_clamp_[1],
               (double)this->geo_abias_clamp_[2], this->geo_abias_max_);
        fflush(stdout);
      }

      this->imu_calibrated = true;

    }

  } else {

    double dt = imu->header.stamp.toSec() - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu->header.stamp.toSec();
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    // NOTE: the geometric-observer PREDICT is NOT done here (real time) anymore.
    // It is done scan-windowed in getNextPose() via propagateStateScanWindow(),
    // so the state never runs ahead of the scan being processed under lag. This
    // callback now only buffers IMU.

  }

}

void dlio::OdomNode::getNextPose() {

  // Geometric-observer PREDICT, scan-windowed: advance the state to THIS scan's
  // time using only the IMU within (prev_scan_stamp, scan_stamp], so it is never
  // ahead of the scan (see propagateStateScanWindow). Gated on first_opt_done to
  // match the upstream behavior (no propagation before the first optimization).
  if (this->geo.first_opt_done) {
    this->propagateStateScanWindow();
  }

  // Check if the new submap is ready to be used
  this->new_submap_is_ready = (this->submap_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready);

  if (this->new_submap_is_ready && this->submap_hasChanged) {

    // Set the current global submap as the target cloud
    this->gicp.registerInputTarget(this->submap_cloud);

    // Set submap kdtree
    this->gicp.target_kdtree_ = this->submap_kdtree;

    // Set target cloud's normals as submap normals
    this->gicp.setTargetCovariances(this->submap_normals);

    this->submap_hasChanged = false;
  }

  // Align with current submap with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned (boost::make_shared<pcl::PointCloud<PointType>>());
  this->gicp.align(*aligned);

  // Get final transformation in global frame
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;

  // INSTRUMENT ONLY: how far the registration moved this scan, BEFORE any
  // guard_pose correction. Nothing reads it but the [DEGEN] line; it is the
  // registration's own movement, which the exported pose step only approximates
  // (the observer sits between them).
  this->degen_dp_ = (double)(this->T.block<3,1>(0,3) - this->T_prior.block<3,1>(0,3)).norm();

  // Score THIS scan's registration geometry (no-op unless the guard is armed
  // or observing). Must run here: the Hessian belongs to the align() above.
  this->scoreDegeneracy();

  // Optional site 2 — guard the GICP translation increment itself, so the
  // PUBLISHED pose does not take an unobservable jump either. T_corr must move
  // by the same delta: it is what publishToROS transforms the recorded cloud by
  // and what keyframe_transformations stores, so guarding T alone would put the
  // clouds somewhere the poses do not agree with.
  if (this->degen_params_.enabled && this->degen_params_.guard_pose &&
      this->degen_w_.valid && !this->degen_w_.is_identity()) {
    Eigen::Vector3f p_prior = this->T_prior.block<3,1>(0,3);
    Eigen::Vector3f dp      = this->T.block<3,1>(0,3) - p_prior;
    Eigen::Vector3f delta   = dlio::degeneracy::project_observable(dp, this->degen_w_) - dp;
    this->T.block<3,1>(0,3)      += delta;
    this->T_corr.block<3,1>(0,3) += delta;
  }

  // Update next global pose
  // Both source and target clouds are in the global frame now, so tranformation is global
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

  // --- E4 / ARCHITECT A: THE NUDGE ------------------------------------------
  // HERE, and not one statement later. deskewPointcloud() builds scan k+1's
  // prior from lidarPose and geo.prev_vel and it runs BEFORE getNextPose() in
  // the next callback, so a correction written here is already in the next
  // prior; a correction written anywhere downstream is not. Nothing has read
  // the state yet: updateKeyframes(), buildSubmap(), publishPose() and
  // publishToROS() all run after this returns, and every one of them inherits
  // the corrected transform for free.
  this->smootherUpdate();

  // Report AFTER the update, so the computed weights and the applied metres in
  // the same [DEGEN] line belong to the same scan. The [DEGEN] line describes
  // the REGISTRATION, and it still does: Tq/Tp, H6w/H6b, dp6 and corr6 were all
  // captured in scoreDegeneracy(), before the nudge existed for this scan.
  this->logDegeneracy();
  this->logSmoother();

}

// Score the current scan's GICP geometry and report it.
//
// The instrument, not the fix: with degeneracy/observe:=true this changes NO
// arithmetic and only prints, which is how tau_lo/tau_hi get FITTED on a real
// bag instead of guessed. With degeneracy/enabled:=true the weights computed
// here are what updateState() projects its error onto.
void dlio::OdomNode::scoreDegeneracy() {

  if (!this->degen_params_.scoring()) {
    this->degen_w_ = dlio::degeneracy::Weights();   // valid=false, w=(1,1,1)
    return;
  }

  // H is [rot(0..2) | trans(3..5)] in the TARGET (global) frame — the same frame
  // as the observer error, so nothing needs rotating. hasFinalHessian() refuses
  // the scans whose LM never accepted a step, where final_hessian_ still holds
  // the PREVIOUS scan's geometry.
  const Eigen::Matrix<double,6,6>& H = this->gicp.getFinalHessian();
  // Which 3x3 block is the TRANSLATION information lives in degeneracy.h, where
  // a unit test can assert it against a 6x6 whose two blocks differ (picking the
  // rotation block at (0,0) by mistake scores radians, silently).
  const Eigen::Matrix3d Htt = dlio::degeneracy::translation_information(H);

  this->degen_w_ = dlio::degeneracy::degeneracy_weights(
      Htt, this->gicp.hasFinalHessian(), this->gicp.num_correspondences, this->degen_params_);

  // INSTRUMENT ONLY. The rotation block of the SAME Hessian, plus the whole
  // 6x6's condition number. Nothing acts on it -- degen_w_ above is still built
  // from the translation block alone and is the only thing updateState() sees.
  // It is here because the corridor's r_min is a translation number while the
  // onset is a heading error, and a column that does not exist cannot be fitted.
  this->degen_rot_ = dlio::degeneracy::rotation_record(H, this->gicp.hasFinalHessian());

  // E2 -- THE TRUE 6x6, and the registration's own relative pose.
  //
  // Same H, same scan, same validity gate as the two eigen-summary records
  // above; what is added is the 21 free entries of the matrix ITSELF, both as
  // nano_gicp built it (WORLD/LEFT, anchored at the world origin) and
  // re-anchored on the sensor in GTSAM's RIGHT/BODY Pose3 tangent. The second
  // is the quantity a smoother can use; the first is kept so the conversion is
  // CHECKABLE offline against this same line's lam/rlam, rather than trusted.
  //
  // (R, t) is THIS scan's GICP pose `T = T_corr * T_prior`, read here -- after
  // align(), before the optional guard_pose adjustment and before
  // propagateGICP() copies it into lidarPose. That is the pose at which the
  // Hessian was linearised, which is the only pose the adjoint may use.
  const Eigen::Matrix3d R_w = this->T.block<3,3>(0,0).cast<double>();
  const Eigen::Vector3d t_w = this->T.block<3,1>(0,3).cast<double>();
  this->degen_h_ = dlio::degeneracy::hessian_record(
      H, this->gicp.hasFinalHessian(), R_w, t_w,
      this->gicp.getFinalError(), this->gicp.num_correspondences);

  // The registration's own correction to the IMU prior, as a 6-VECTOR rather
  // than the |dp| scalar beside it: T_prior -> T, expressed in T_prior's frame.
  // This is the quantity `innov` is the observer-side counterpart of, and the
  // one a chi-square against the 6x6 above is taken on.
  this->degen_corr6_ = dlio::degeneracy::relative_rot_trans(
      this->T_prior.block<3,3>(0,0).cast<double>(),
      this->T_prior.block<3,1>(0,3).cast<double>(), R_w, t_w);

  // The scan-to-scan GICP relative pose, T_{k-1}^-1 T_k -- the MEASUREMENT a
  // BetweenFactor<Pose3> takes. E0 had to substitute the exported pose chain
  // for it (its S2), which is the observer's output and differs from this by
  // exactly `innov`; this field removes that substitution. Zero and flagged
  // invalid on the first scored scan, which has no predecessor.
  this->degen_dp6_valid_ = this->degen_T_prev_valid_;
  if (this->degen_dp6_valid_) {
    this->degen_dp6_ = dlio::degeneracy::relative_rot_trans(
        this->degen_T_prev_.block<3,3>(0,0).cast<double>(),
        this->degen_T_prev_.block<3,1>(0,3).cast<double>(), R_w, t_w);
  } else {
    this->degen_dp6_.setZero();
  }
  this->degen_T_prev_ = this->T;
  this->degen_T_prev_valid_ = true;

  const dlio::degeneracy::Weights& W = this->degen_w_;
  ++this->degen_scans_;
  if (!W.valid) ++this->degen_invalid_;
  if (W.degenerate()) ++this->degen_degenerate_;

  this->degen_flag_       = W.degenerate();
  this->degen_ratio_min_  = W.valid ? W.ratio_min()  : 1.;
  this->degen_w_min_      = W.w_min();

}

// E2 -- number formatting for the [DEGEN] line's vector fields.
//
// %.17g, not %.6g: these are not a human-readable summary, they are the INPUT
// to an offline solve. A Hessian entry rounded to six figures cannot be checked
// against the eigenvalues printed beside it, and the whole point of dumping the
// matrix is that the reconstruction E0 had to use can be compared with the
// truth. 17 significant digits round-trips an IEEE double exactly.
//
// Writes into a caller-owned buffer and is called OUTSIDE the print lock, so
// the locked region still holds exactly one printf and one flush.
static void fmt_doubles(char* buf, size_t n, const double* v, int count) {
  size_t off = 0;
  for (int i = 0; i < count && off < n; ++i) {
    const int w = snprintf(buf + off, n - off, i ? ",%.17g" : "%.17g", v[i]);
    if (w < 0) break;
    off += (size_t)w;
  }
  if (n) buf[n - 1] = '\0';
}

// Report the scan just scored. Called AFTER updateState() so `removed` is THIS
// scan's applied amount and not the previous one's — the whole point of the
// pairing is that the two numbers describe the same scan.
void dlio::OdomNode::logDegeneracy() {

  if (!this->degen_params_.scoring()) return;

  const dlio::degeneracy::Weights& W = this->degen_w_;
  const int every = std::max(1, this->degen_params_.log_every);
  if ((this->degen_scans_ % every) == 0) {
    const Eigen::Vector3d u = W.U.col(0);
    // The thresholds are printed on every line ON PURPOSE: an r/w pair cannot be
    // read without them, and a line copied into a report with the band left
    // implicit is how a forced-threshold smoke run gets mistaken for what the
    // defaults do. `rm` is this scan's metres, `RM` the run's cumulative total.
    // Under print_mutex_ so a flush cannot land inside a status-banner line.
    // E2 -- format the vector fields BEFORE taking the lock, so the locked
    // region is still exactly one printf and a flush. 26 chars is the widest a
    // %.17g double plus its separator can be.
    char h6w[21 * 26], h6b[21 * 26], dp6[6 * 26], corr6[6 * 26], innov6[6 * 26];
    fmt_doubles(h6w, sizeof(h6w), this->degen_h_.Hw, 21);
    fmt_doubles(h6b, sizeof(h6b), this->degen_h_.Hb, 21);
    fmt_doubles(dp6, sizeof(dp6), this->degen_dp6_.data(), 6);
    fmt_doubles(corr6, sizeof(corr6), this->degen_corr6_.data(), 6);
    fmt_doubles(innov6, sizeof(innov6), this->degen_innov6_.data(), 6);
    const int dp6_valid = this->degen_dp6_valid_ ? 1 : 0;
    // The pose the adjoint was taken at: THIS scan's GICP pose, captured in
    // scoreDegeneracy() before the optional guard_pose adjustment. Printed so a
    // reader can rebuild Ad(T) and check H6b against H6w from this line alone,
    // with no join to any other file.
    const Eigen::Quaterniond Tq(this->degen_T_prev_.block<3,3>(0,0).cast<double>());
    const Eigen::Vector3d    Tp = this->degen_T_prev_.block<3,1>(0,3).cast<double>();
    std::lock_guard<std::mutex> print_lock(this->print_mutex_);
    // The new fields are APPENDED, never inserted: every parser written against
    // the first three runs' logs (proof_data/analyse_run.py, aa/fine.py) matches
    // this line with a regex that ends at inv=, and must keep matching.
    printf("[DEGEN] t=%.4f valid=%d ncorr=%d lam=%.6g %.6g %.6g r=%.6f %.6f %.6f "
           "w=%.4f %.4f %.4f u_min=(%.4f,%.4f,%.4f) weak=%d rm=%.6f RM=%.4f "
           "tau=%.4f/%.4f wfloor=%.3f mode=%s "
           "n=%ld deg=%ld app=%ld inv=%ld "
           "q_min=%.6g info=%.4f/%.4f innov=%.4f dp=%.4f "
           "imu_rx=%ld imu_hz=%.3f imu_dtmax=%.4f imu_gaps=%ld "
           "smsz=%d smkcc=%d smref=%ld smdmax=%.2f smdkcc=%.2f "
           "kfage=%.2f kfn=%ld kfmax=%.2f kfmin=%.3f "
           "abmax=%.4f,%.4f,%.4f abmrg=%.4f "
           "rvalid=%d rlam=%.6g %.6g %.6g rr=%.6f ru_min=(%.4f,%.4f,%.4f) cond6=%.6g "
           "hvalid=%d ncorr_raw=%d ferr=%.17g "
           "Tq=%.17g,%.17g,%.17g,%.17g Tp=%.17g,%.17g,%.17g "
           "H6w=%s H6b=%s dp6v=%d dp6=%s corr6=%s innov6=%s "
           "units=lam:corr_count;q:corr_count_per_corr;rm:m;innov:m;dp:m;"
           "imu_hz:Hz;imu_dtmax:s;smsz:keyframes;smkcc:indices;smref:lists;"
           "smdmax:m;smdkcc:m;kfage:s;kfn:keyframes;kfmax:s;kfmin:m;"
           "abmax:m_per_s2;abmrg:m_per_s2;rlam:corr_count_times_m2;rr:ratio;"
           "cond6:MIXED_UNITS_relative_only;"
           // --- E2 ---------------------------------------------------------
           // H6w / H6b are the SAME 6x6 in two charts. 21 = the free entries of
           // a symmetric 6x6, packed ROW-MAJOR OVER THE UPPER TRIANGLE:
           // (0,0)(0,1)..(0,5)(1,1)..(1,5)(2,2)..(2,5)(3,3)(3,4)(3,5)(4,4)(4,5)(5,5).
           // Block order is [rot 0..2 | trans 3..5] in BOTH.
           //   H6w  exactly what nano_gicp accumulated: a LEFT perturbation in
           //        the WORLD frame with the rotation Jacobian taken about the
           //        WORLD ORIGIN, so its rotation block carries the squared
           //        distance from that origin. lam / rlam on this same line are
           //        the eigenvalues of its (3,3) and (0,0) blocks -- which is
           //        how a reader checks that this field is what it says it is.
           //   H6b  the same information in gtsam::Pose3's RIGHT / BODY tangent
           //        at the pose Tq/Tp: Ad(T)^T H6w Ad(T). The cross block
           //        cancels the world-origin lever arm exactly, so H6b's
           //        rotation block is SENSOR-anchored -- the attitude
           //        information of the scene, not of where the scene sits
           //        relative to (0,0,0).
           // Both are correspondence counts, not informations in m^-2: PLANE
           // regularisation forces every point covariance to (1,1,1e-3) and the
           // scalar that converts them is fitted offline, never here.
           // The three 6-vectors share ONE chart, the DECOUPLED
           // [rotvec(3) rad ; translation(3) m]: v = (log_SO3(R_A^T R_B),
           // R_A^T (t_B - t_A)) for a relative pose A^-1 B. Rebuild it exactly
           // with Pose3(Rot3::Expmap(v[0:3]), v[3:6]). It is NOT gtsam's
           // Pose3::Logmap, which carries V(omega)^-1 on the translation half.
           "H6w:UPPER21_ROWMAJOR_rot0to2_trans3to5_WORLD_LEFT_ORIGIN_ANCHORED;"
           "H6b:UPPER21_ROWMAJOR_rot0to2_trans3to5_BODY_RIGHT_SENSOR_ANCHORED;"
           "H6blocks:rot_rot:corr_count_times_m2,rot_trans:corr_count_times_m,"
           "trans_trans:corr_count;"
           "Tq:quaternion_wxyz_world_from_body;Tp:m;"
           "dp6:DECOUPLED_rotvec_rad_then_m,prev_GICP_pose_to_this_GICP_pose;"
           "corr6:DECOUPLED_rotvec_rad_then_m,T_prior_to_T_gicp;"
           "innov6:DECOUPLED_rotvec_rad_then_m,state_to_lidarPose;"
           "ferr:sum_mahalanobis_sq_error_at_accepted_LM_step;"
           "ncorr_raw:corr_count_UNGATED\n",
           this->scan_stamp, (int)W.valid, W.ncorr,
           W.lambda(0), W.lambda(1), W.lambda(2),
           W.ratio(0), W.ratio(1), W.ratio(2),
           W.w(0), W.w(1), W.w(2),
           u(0), u(1), u(2), W.weak,
           (double)this->degen_removed_, (double)this->degen_removed_total_,
           this->degen_params_.min_ratio, this->degen_params_.full_ratio,
           this->degen_params_.min_weight,
           this->degen_params_.enabled ? "guard" : "observe",
           (long)this->degen_scans_, (long)this->degen_degenerate_,
           (long)this->degen_applied_, (long)this->degen_invalid_,
           W.info(0),
           this->degen_params_.min_info, this->degen_params_.full_info,
           (double)this->degen_innov_, (double)this->degen_dp_,
           (long)this->imu_rx_n_, (double)this->imu_rx_hz_,
           (double)this->imu_rx_maxdt_, (long)this->imu_rx_gaps_,
           (int)this->submap_size_, (int)this->submap_kcc_added_,
           (long)this->submap_short_refused_,
           (double)this->submap_dmax_, (double)this->submap_dkcc_,
           (double)this->keyframe_age_last_s_, (long)this->keyframe_age_fired_,
           this->keyframe_age_.max_age_s, this->keyframe_age_travel_m_,
           (double)this->geo_abias_clamp_[0], (double)this->geo_abias_clamp_[1],
           (double)this->geo_abias_clamp_[2], this->geo_abias_margin_,
           (int)this->degen_rot_.valid,
           this->degen_rot_.lambda(0), this->degen_rot_.lambda(1),
           this->degen_rot_.lambda(2), this->degen_rot_.ratio_min(),
           this->degen_rot_.u_min(0), this->degen_rot_.u_min(1),
           this->degen_rot_.u_min(2), this->degen_rot_.cond6,
           (int)this->degen_h_.valid, this->degen_h_.ncorr, this->degen_h_.ferr,
           Tq.w(), Tq.x(), Tq.y(), Tq.z(), Tp(0), Tp(1), Tp(2),
           h6w, h6b, dp6_valid, dp6, corr6, innov6);
    fflush(stdout);
    if (this->degen_h_.valid) ++this->degen_h6_written_;
  }

  // SILENT NO-OP LAW for the E2 record itself. COMPUTED = scans scored;
  // APPLIED = lines that actually carried a 6x6. The way this instrument fails
  // silently is not a wrong number, it is hvalid=0 on every line -- an
  // unconverged LM leaves no fresh Hessian and hessian_record() refuses, so a
  // run can score 11,000 scans and dump nothing while every other column looks
  // healthy. That is indistinguishable from a working dump until somebody tries
  // to fit on it, which is a 40-minute bag replay too late.
  if (!this->degen_h6_noop_reported_ && this->degen_scans_ > 1000 &&
      this->degen_h6_written_ == 0) {
    this->degen_h6_noop_reported_ = true;
    fprintf(stderr,
            "[DEGEN][ERROR] %ld scans scored and NOT ONE carried a 6x6 "
            "(hvalid=0 on every line). getFinalHessian() is never fresh: check "
            "that nano_gicp's LM is accepting steps (hasFinalHessian()), and do "
            "not fit anything on this run.\n",
            (long)this->degen_scans_.load());
    fflush(stderr);
  }

  // SILENT NO-OP LAW, both directions.
  //
  // (a) COMPUTED > 0 while APPLIED == 0: the guard scored degenerate scans and
  //     changed nothing. Hard error.
  if (this->degen_params_.enabled && !this->degen_noop_reported_ &&
      this->degen_degenerate_ > 100 && this->degen_applied_ == 0) {
    this->degen_noop_reported_ = true;
    fprintf(stderr,
            "[DEGEN][ERROR] guard enabled and %ld scans scored DEGENERATE but the "
            "observer error was never modified (applied=0). The guard is a no-op — "
            "check that dlio/odom/gicp/degeneracy/enabled reached the node.\n",
            (long)this->degen_degenerate_);
    fflush(stderr);
  }

  // (b) The COMPLEMENTARY no-op, which is the more probable one: armed, running,
  //     and nothing ever scored degenerate — because min_ratio is below the
  //     Hessian's own floor, max_weak_dirs is 0, min_corr refuses every scan, or
  //     (most likely) the yaml never reached the node, since these parameters can
  //     only arrive through a SHALLOW-merged JSON blob. deg=0/N and a silent run
  //     look exactly like a healthy scene, so say which it is. 1,000 scans =
  //     100 s of a 10 Hz sensor, well past the Sandland corridor entry at 137.9 s
  //     only if the corridor is there at all — hence the wording.
  if (this->degen_params_.enabled && !this->degen_blind_reported_ &&
      this->degen_scans_ > 1000 && this->degen_degenerate_ == 0) {
    this->degen_blind_reported_ = true;
    fprintf(stderr,
            "[DEGEN][WARN] guard enabled for %ld scans (%ld refused) and NOT ONE was "
            "scored degenerate. Either the geometry never degenerated, or the guard "
            "cannot see it: check the r column against min_ratio=%.4f/full_ratio=%.4f, "
            "the q_min column against min_info=%.4f/full_info=%.4f (0/0 = the absolute "
            "test is off), max_weak_dirs=%d, min_corr=%d, and that these values reached "
            "the node (rosparam dump /robot/dlio_odom), not just the recipe row.\n",
            (long)this->degen_scans_, (long)this->degen_invalid_,
            this->degen_params_.min_ratio, this->degen_params_.full_ratio,
            this->degen_params_.min_info, this->degen_params_.full_info,
            this->degen_params_.max_weak_dirs, this->degen_params_.min_corr);
    fflush(stderr);
  }
}


// ===========================================================================
//  E4 / ARCHITECT A -- THE NUDGE.  include/dlio/smoother.h has the design.
// ===========================================================================

// One scan in, one solve, and the SIX WRITES.
//
// The correction has to reach the thing the registration is measured AGAINST,
// not only the state: the anchor (lidarPose, geo.prev_vel) and the map (the
// submap keyframes). A correction written into state alone survives exactly one
// scan and reads as a null result -- which is this design's most probable
// defect and the reason write_back() reports what it did.
void dlio::OdomNode::smootherUpdate() {

  if (!this->smoother_) return;

  dlio::smoother::ScanInput in;
  in.stamp      = this->scan_stamp;
  in.prev_stamp = this->prev_scan_stamp;
  // The registration's own pose for this scan, T = T_corr * T_prior, read here
  // -- after align(), after propagateGICP(), after updateState(), and before
  // anything downstream has seen it. The guard's optional pose adjustment
  // cannot have moved it: arming the guard and the smoother together is refused
  // in the constructor, because two actions on one scan are unattributable.
  in.T_gicp  = this->T.cast<double>();
  in.H6b     = dlio::degeneracy::from_upper21(this->degen_h_.Hb);
  in.h_valid = this->degen_h_.valid;
  in.ncorr   = this->degen_h_.ncorr;
  in.ferr    = this->degen_h_.ferr;
  in.dp6     = this->degen_dp6_;
  in.dp6_valid = this->degen_dp6_valid_;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    in.v_world = this->state.v.lin.w.cast<double>();
  }
  {
    // The RAW samples of this scan interval. p50 64 at 640 Hz.
    std::lock_guard<std::mutex> lk(this->mtx_raw_imu_);
    in.imu.reserve(96);
    for (const auto& r : this->raw_imu_buffer_) {
      if (r.stamp > in.prev_stamp && r.stamp <= in.stamp) {
        dlio::smoother::ImuSample sm;
        sm.stamp = r.stamp;
        sm.accel = r.accel;
        sm.gyro  = r.gyro;
        in.imu.push_back(sm);
      }
    }
  }

  this->smoother_sol_ = this->smoother_->update(in);
  ++this->smoother_scans_;
  this->smoother_solve_ms_.push_back(this->smoother_sol_.solve_ms);
  if (this->smoother_sol_.solve_ms > this->smoother_solve_ms_max_) {
    this->smoother_solve_ms_max_ = this->smoother_sol_.solve_ms;
  }

  // FREEZE ON MARGINALISATION. Every keyframe whose scan left the window on
  // this update is frozen here, so a later delta offered to it is REFUSED and
  // counted rather than silently applied to a pose the graph no longer owns.
  if (!this->smoother_sol_.kf_frozen.empty()) {
    std::lock_guard<std::mutex> lk(this->kf_delta_mutex_);
    for (int i : this->smoother_sol_.kf_frozen) {
      if (i >= 0 && i < (int)this->kf_frozen_.size()) this->kf_frozen_[i] = 1;
    }
  }

  // --- THE SIX WRITES, all under geo.mtx, all or none --------------------
  dlio::smoother::WriteTargets t;
  {
    std::lock_guard<std::mutex> lock(this->geo.mtx);
    t.T             = &this->T;
    t.T_corr        = &this->T_corr;
    t.T_prior       = &this->T_prior;
    t.lidar_p       = &this->lidarPose.p;
    t.lidar_q       = &this->lidarPose.q;
    t.state_p       = &this->state.p;
    t.state_q       = &this->state.q;
    t.v_world       = &this->state.v.lin.w;
    t.v_body        = &this->state.v.lin.b;
    t.geo_prev_p    = &this->geo.prev_p;
    t.geo_prev_q    = &this->geo.prev_q;
    t.geo_prev_vel  = &this->geo.prev_vel;
    t.b_accel_bl    = &this->state.b.accel;
    t.b_gyro_bl     = &this->state.b.gyro;
    t.kf            = this->smoother_params_.keyframe_writeback
                          ? (dlio::smoother::KeyframeSink*)this->kf_sink_.get()
                          : nullptr;
    t.kf_dirty_trans_m = this->smoother_params_.kf_dirty_trans_m;
    t.kf_dirty_rot_deg = this->smoother_params_.kf_dirty_rot_deg;
    t.R_bl_imu      = this->extrinsics.baselink2imu.R;
    this->smoother_rep_ = dlio::smoother::write_back(t, this->smoother_sol_,
                                                     this->smoother_params_.alpha);
  }
  this->smoother_ledger_.note(this->smoother_sol_, this->smoother_rep_);
}

// The [SMOOTH] record. One line per scan, its own units= token, and the
// silent-no-op law underneath it. The run must be able to SHOW it did
// something: solve time, window size, the correction it applied in metres and
// degrees, the marginal's own spectrum, alpha, and the two factor residuals
// side by side so a reader can see WHICH one moved.
void dlio::OdomNode::logSmoother() {

  if (!this->smoother_) return;

  const dlio::smoother::Solution& S = this->smoother_sol_;
  const dlio::smoother::WriteBackReport& W = this->smoother_rep_;
  const int every = std::max(1, this->smoother_params_.log_every);
  if ((this->smoother_scans_ % every) == 0) {
    std::lock_guard<std::mutex> print_lock(this->print_mutex_);
    printf("[SMOOTH] t=%.4f k=%ld valid=%d solved=%d solve_ms=%.3f "
           "win_vars=%d win_fac=%d lm=%d lag_s=%.2f "
           "alpha=%.4f corr_m=%.6f corr_deg=%.6f app_m=%.6f app_deg=%.6f "
           "groups=%d kfw=%d kffz=%d "
           "mcov=%.6g,%.6g,%.6g,%.6g,%.6g,%.6g "
           "r_imu=%.6g r_reg=%.6g regf=%d hvalid=%d ncorr=%d "
           "imu_n=%d imu_gaps=%d floor=%d psd=%d exc=%d reseats=%ld "
           "s_rot=%.9g s_trans=%.9g "
           "n=%ld computed=%ld applied=%ld kf_applied=%ld kf_refused=%ld "
           "units=solve_ms:ms;corr_m:m;corr_deg:deg;app_m:m;app_deg:deg;"
           "mcov:VARIANCE_diag_of_the_marginal_on_X_k_in_gtsam_Pose3_tangent_"
           "rot0to2_rad2_trans3to5_m2;r_imu:whitened_factor_error_CombinedImu;"
           "r_reg:whitened_factor_error_BetweenPose3_after_Huber;"
           "s_rot:rad^-2_per_corr_count_times_m2;s_trans:m^-2_per_corr_count;"
           "win_vars:variables_in_the_fixed_lag_window;win_fac:factors;"
           "groups:WRITE_GROUPS_of_6_T_lidarPose_state_geoprev_bias_keyframes;"
           "kfw:in_window_keyframes_moved_THIS_scan;"
           "kffz:keyframes_that_LEFT_the_window_THIS_scan;"
           "reseats:fixed_lag_windows_REBUILT_after_a_failed_update_"
           "cumulative_a_run_with_any_is_not_a_clean_arm;"
           "alpha:BLEND_never_fitted_0_runs_and_writes_nothing\n",
           this->scan_stamp, (long)this->smoother_scans_ - 1, (int)S.valid,
           (int)S.solved_this_scan, S.solve_ms, S.window_vars, S.window_factors,
           S.lm_iterations, this->smoother_params_.lag_s,
           this->smoother_params_.alpha, W.corr_m, W.corr_deg, W.applied_m,
           W.applied_deg, W.groups_written, W.kf_written,
           (int)S.kf_frozen.size(), S.marg_cov[0], S.marg_cov[1], S.marg_cov[2],
           S.marg_cov[3], S.marg_cov[4], S.marg_cov[5], S.resid_imu, S.resid_reg,
           (int)this->degen_dp6_valid_, (int)this->degen_h_.valid,
           this->degen_h_.ncorr, S.imu_samples, S.imu_gaps,
           (int)S.floor_binding, (int)S.psd_projected, (int)S.update_exception,
           (long)S.reseats,
           this->smoother_params_.info_scale_rot,
           this->smoother_params_.info_scale_trans,
           this->smoother_ledger_.scans, this->smoother_ledger_.computed,
           this->smoother_ledger_.applied,
           (long)this->smoother_kf_applied_, (long)this->smoother_kf_refused_);
    fflush(stdout);
  }

  // --- SILENT NO-OP LAW ----------------------------------------------------
  // COMPUTED = scans on which the smoother produced a correction above 1 mm.
  // APPLIED  = scans on which the state actually moved. computed > 0 &&
  // applied == 0 is a HARD ERROR -- unless alpha == 0, where applied == 0 is
  // the POINT and the law binds the other way: the smoother must still have
  // COMPUTED, or the A/A control is measuring nothing.
  if (!this->smoother_noop_reported_) {
    const std::string v =
        this->smoother_ledger_.violation(this->smoother_params_.alpha);
    if (!v.empty()) {
      this->smoother_noop_reported_ = true;
      fprintf(stderr, "[SMOOTH][ERROR] %s\n", v.c_str());
      fflush(stderr);
    }
  }
  // A keyframe delta offered to a FROZEN keyframe is a broken invariant, not a
  // skip: the window and the map disagree about which poses are still owned.
  if (this->smoother_rep_.kf_frozen_refused > 0) {
    this->smoother_kf_refused_ += this->smoother_rep_.kf_frozen_refused;
    fprintf(stderr,
            "[SMOOTH][ERROR] %d delta(s) were offered to a keyframe that has "
            "already left the smoother's window (frozen). A marginalised pose "
            "must never move again -- that is what makes the kd-tree rebuild "
            "bounded. Scan k=%ld.\n",
            this->smoother_rep_.kf_frozen_refused,
            (long)this->smoother_scans_ - 1);
    fflush(stderr);
  }
}

// --- the keyframe sink ------------------------------------------------------

bool dlio::OdomNode::KfSink::poseOf(int kf_index, Eigen::Matrix4d* out) const {
  std::lock_guard<std::mutex> lk(this->node->kf_delta_mutex_);
  if (kf_index < 0 || kf_index >= (int)this->node->kf_pose_now_.size()) return false;
  if (out) *out = this->node->kf_pose_now_[kf_index];
  return true;
}

bool dlio::OdomNode::KfSink::queueDelta(int kf_index, const Eigen::Matrix4d& delta,
                                        const Eigen::Matrix4d& pose_new) {
  std::lock_guard<std::mutex> lk(this->node->kf_delta_mutex_);
  if (kf_index < 0 || kf_index >= (int)this->node->kf_frozen_.size()) return false;
  if (this->node->kf_frozen_[kf_index]) return false;   // FROZEN: refuse, loudly
  // Deltas COMPOSE: the submap builder is asynchronous and may not have drained
  // the previous one yet, so a second correction to the same keyframe in the
  // same build interval must multiply onto the first, never replace it.
  if (this->node->kf_has_delta_[kf_index]) {
    this->node->kf_pending_delta_[kf_index] =
        delta * this->node->kf_pending_delta_[kf_index];
  } else {
    this->node->kf_pending_delta_[kf_index] = delta;
    this->node->kf_has_delta_[kf_index] = 1;
  }
  this->node->kf_pose_now_[kf_index] = pose_new;
  return true;
}

void dlio::OdomNode::KfSink::markSubmapDirty() {
  this->node->smoother_submap_dirty_ = true;
}

// Applied on the submap-build thread, BEFORE the new keyframes are baked and
// before buildSubmap() composes. Two cases, and confusing them would double-
// apply the correction:
//   i  < num_processed_keyframes : the cloud is ALREADY in the world frame
//                                  (buildKeyframesAndSubmap baked it once and
//                                  dropped the raw), so the delta moves the
//                                  cloud and the covariances directly.
//   i >= num_processed_keyframes : the cloud is still the raw current_scan and
//                                  keyframe_transformations[i] is what will
//                                  bake it -- so the delta goes into THAT and
//                                  the cloud is left alone.
// A world->world delta is exact and needs no raw cloud: it is the same
// std::transform buildKeyframesAndSubmap already runs, at one 4x4 per point.
void dlio::OdomNode::applyKeyframeDeltas() {

  if (!this->smoother_ || !this->smoother_params_.keyframe_writeback) return;

  std::vector<int> idx;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> del;
  {
    std::lock_guard<std::mutex> lk(this->kf_delta_mutex_);
    for (std::size_t i = 0; i < this->kf_has_delta_.size(); ++i) {
      if (!this->kf_has_delta_[i]) continue;
      idx.push_back((int)i);
      del.push_back(this->kf_pending_delta_[i]);
      this->kf_has_delta_[i] = 0;
      this->kf_pending_delta_[i] = Eigen::Matrix4d::Identity();
    }
    this->smoother_submap_dirty_ = false;
  }
  if (idx.empty()) return;

  long applied = 0;
  {
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    for (std::size_t n = 0; n < idx.size(); ++n) {
      const int i = idx[n];
      if (i < 0 || i >= (int)this->keyframes.size()) continue;
      const Eigen::Matrix4d& Dd = del[n];
      const Eigen::Matrix4f D = Dd.cast<float>();
      const Eigen::Matrix3f DR = D.block<3, 3>(0, 0);

      Eigen::Vector3f p = this->keyframes[i].first.first;
      Eigen::Quaternionf q = this->keyframes[i].first.second;
      Eigen::Vector3f p2 = DR * p + D.block<3, 1>(0, 3);
      Eigen::Quaternionf q2 = Eigen::Quaternionf(DR) * q;
      q2.normalize();
      this->keyframes[i].first = std::make_pair(p2, q2);
      this->keyframe_transformations[i] = D * this->keyframe_transformations[i];

      if (i < this->num_processed_keyframes) {
        pcl::PointCloud<PointType>::Ptr moved
            (boost::make_shared<pcl::PointCloud<PointType>>());
        pcl::transformPointCloud(*this->keyframes[i].second, *moved, D);
        this->keyframes[i].second = moved;
        std::shared_ptr<nano_gicp::CovarianceList> cov
            (std::make_shared<nano_gicp::CovarianceList>(
                this->keyframe_normals[i]->size()));
        std::transform(this->keyframe_normals[i]->begin(),
                       this->keyframe_normals[i]->end(), cov->begin(),
                       [&Dd](Eigen::Matrix4d c) { return Dd * c * Dd.transpose(); });
        this->keyframe_normals[i] = cov;
      }
      ++applied;
    }
  }

  if (applied > 0) {
    this->smoother_kf_applied_ = this->smoother_kf_applied_.load() + applied;
    // MARK THE SUBMAP DIRTY. buildSubmap() only recomposes when the INDEX SET
    // changed; a member that MOVED leaves the set identical, so without this
    // the registration keeps aiming at the unmoved cloud and the correction is
    // undone by the very measurement it was meant to reweigh. Clearing prev
    // forces the recomposition and, with it, the kd-tree rebuild -- lazily, on
    // the next align, which is the existing cost at a higher frequency and not
    // a new one.
    this->submap_kf_idx_prev.clear();
    std::lock_guard<std::mutex> print_lock(this->print_mutex_);
    printf("[SMOOTH] submap DIRTY: %ld in-window keyframe(s) moved, submap will "
           "be recomposed and the kd-tree rebuilt on the next align "
           "units=keyframes\n", applied);
    fflush(stdout);
  }
}

bool dlio::OdomNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::OdomNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

    prev_imu_it = imu_it;

  }

  return imu_se3;

}

void dlio::OdomNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::OdomNode::propagateState(const ImuMeas& m) {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = m.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(m.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = m.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = m.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

// Geometric-observer PREDICT, scan-windowed. The upstream design propagated the
// state on EVERY IMU as it arrived (real time). Under any processing lag (e.g.
// flushing a buffered backlog when a bag is played faster than DLIO can run),
// the IMU floods ahead while the lidar callback is still on an OLD scan, so the
// state runs far past it -> updateState() sees a huge error vs the (correct) old
// scan pose -> the geometric observer diverges -> degenerate clouds ("Low number
// of points"). Here we instead advance the state using ONLY the buffered IMU
// inside (prev_scan_stamp, scan_stamp] -- the window of the scan we're about to
// fuse -- so the state is always exactly at the scan being processed, never
// ahead. At real time (no lag) this consumes the same samples as before, so the
// result is identical; under lag it stays consistent. This makes the output a
// function of the bag, independent of playback speed.
void dlio::OdomNode::propagateStateScanWindow() {

  std::vector<ImuMeas> window;
  {
    std::lock_guard<std::mutex> imu_lock( this->mtx_imu );
    // imu_buffer is push_front (front = newest); rbegin..rend is oldest..newest.
    for (auto it = this->imu_buffer.rbegin(); it != this->imu_buffer.rend(); ++it) {
      if (it->stamp > this->prev_scan_stamp && it->stamp <= this->scan_stamp) {
        window.push_back(*it);
      }
    }
  }
  for (const auto& m : window) {
    this->propagateState(m);
  }
}

void dlio::OdomNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;

  // INSTRUMENT ONLY: the RAW observer innovation, recorded before the guard can
  // touch it and whether or not the guard is armed. This is the quantity
  // innov_max_m gates, and no run before this one logged it - the A/A analysis
  // had to infer it from exported pose steps because the banner's Position {W}
  // is the same series as the exported pose, not the disagreement.
  this->degen_innov_ = (double)err.norm();

  // E2: the SAME innovation as a 6-VECTOR, so the rotational half stops being
  // invisible. |err| is a translation norm; the onset of this bag's divergence
  // is a heading error, and a translation-only instrument is structurally blind
  // to it (the reason the rotation block was added at all). Chart: rotation
  // vector of qhat^-1 qin, then (pin - state.p) rotated into the STATE frame --
  // the same decoupled [rotvec ; translation] chart every other 6-vector on the
  // line uses. Recorded before the guard, like the scalar above.
  this->degen_innov6_ = dlio::degeneracy::relative_rot_trans(
      qhat.toRotationMatrix().cast<double>(), this->state.p.cast<double>(),
      qin.toRotationMatrix().cast<double>(), pin.cast<double>());

  // --- GICP degeneracy guard ------------------------------------------------
  // Along a direction the scan cannot measure, `err` is not evidence: it is the
  // un-correctable residual of an unobservable DOF. updateState drives THREE
  // integrators off it (accel bias via Kab, position via Kp, velocity via Kv),
  // so clamping any one of them just moves the runaway to the next. Projecting
  // err onto the observable subspace HERE closes all three at once.
  // Zero arithmetic change when the guard is off: project_observable returns its
  // argument bitwise for w = (1,1,1) and for an unscored scan.
  if (this->degen_params_.enabled) {
    const Eigen::Vector3f err_raw = err;
    err = dlio::degeneracy::project_observable(err, this->degen_w_);
    err = dlio::degeneracy::clamp_innovation(err, this->degen_params_.innov_max_m);
    const double removed = (err_raw - err).norm();
    this->degen_removed_ = removed;
    // Cumulative metres of correction authority removed — the number that says
    // whether the guard did too much or too little over the whole run.
    this->degen_removed_total_ = this->degen_removed_total_.load() + removed;
    if (removed > 0.) ++this->degen_applied_;
  } else {
    this->degen_removed_ = 0.;
  }
  // --- end guard ------------------------------------------------------------

  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  // N57: the bound is PER AXIS and comes from dlio/repairs.h. With
  // abias_margin = 0 all three entries equal the configured abias_max and this
  // is the scalar clamp DLIO has always applied, to the bit.
  const Eigen::Vector3f abias_max = this->geo_abias_clamp_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max.array()).max(-abias_max.array());

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::Imu::Ptr dlio::OdomNode::transformImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {

  sensor_msgs::Imu::Ptr imu (new sensor_msgs::Imu);

  // Copy header
  imu->header = imu_raw->header;

  static double prev_stamp = imu->header.stamp.toSec();
  double dt = imu->header.stamp.toSec() - prev_stamp;
  prev_stamp = imu->header.stamp.toSec();
  
  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::OdomNode::computeMetrics() {
  this->computeSpaciousness();
  this->computeDensity();
}

void dlio::OdomNode::computeSpaciousness() {

  // compute range of points
  std::vector<float> ds;

  for (int i = 0; i < this->original_scan->points.size(); i++) {
    float d = std::sqrt(pow(this->original_scan->points[i].x, 2) +
                        pow(this->original_scan->points[i].y, 2));
    ds.push_back(d);
  }

  // median
  std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
  float median_curr = ds[ds.size()/2];
  static float median_prev = median_curr;
  float median_lpf = 0.95*median_prev + 0.05*median_curr;
  median_prev = median_lpf;

  // push
  this->metrics.spaciousness.push_back( median_lpf );

}

void dlio::OdomNode::computeDensity() {

  float density;

  if (!this->geo.first_opt_done) {
    density = 0.;
  } else {
    density = this->gicp.source_density_;
  }

  static float density_prev = density;
  float density_lpf = 0.95*density_prev + 0.05*density;
  density_prev = density_lpf;

  this->metrics.density.push_back( density_lpf );

}

void dlio::OdomNode::computeConvexHull() {

  // at least 4 keyframes for convex hull
  if (this->num_processed_keyframes < 4) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the convex hull of the point cloud
  this->convex_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the convex hull
  pcl::PointCloud<PointType>::Ptr convex_points =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->convex_hull.reconstruct(*convex_points);

  pcl::PointIndices::Ptr convex_hull_point_idx = pcl::PointIndices::Ptr (boost::make_shared<pcl::PointIndices>());
  this->convex_hull.getHullPointIndices(*convex_hull_point_idx);

  this->keyframe_convex.clear();
  for (int i=0; i<convex_hull_point_idx->indices.size(); ++i) {
    this->keyframe_convex.push_back(convex_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::computeConcaveHull() {

  // at least 5 keyframes for concave hull
  if (this->num_processed_keyframes < 5) {
    return;
  }

  // create a pointcloud with points at keyframes
  pcl::PointCloud<PointType>::Ptr cloud =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());

  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    PointType pt;
    pt.x = this->keyframes[i].first.first[0];
    pt.y = this->keyframes[i].first.first[1];
    pt.z = this->keyframes[i].first.first[2];
    cloud->push_back(pt);
  }
  lock.unlock();

  // calculate the concave hull of the point cloud
  this->concave_hull.setInputCloud(cloud);

  // get the indices of the keyframes on the concave hull
  pcl::PointCloud<PointType>::Ptr concave_points =
    pcl::PointCloud<PointType>::Ptr (boost::make_shared<pcl::PointCloud<PointType>>());
  this->concave_hull.reconstruct(*concave_points);

  pcl::PointIndices::Ptr concave_hull_point_idx = pcl::PointIndices::Ptr (boost::make_shared<pcl::PointIndices>());
  this->concave_hull.getHullPointIndices(*concave_hull_point_idx);

  this->keyframe_concave.clear();
  for (int i=0; i<concave_hull_point_idx->indices.size(); ++i) {
    this->keyframe_concave.push_back(concave_hull_point_idx->indices[i]);
  }

}

void dlio::OdomNode::updateKeyframes() {

  // calculate difference in pose and rotation to all poses in trajectory
  float closest_d = std::numeric_limits<float>::infinity();
  int closest_idx = 0;
  int keyframes_idx = 0;

  int num_nearby = 0;

  for (const auto& k : this->keyframes) {

    // calculate distance between current pose and pose in keyframes
    float delta_d = sqrt( pow(this->state.p[0] - k.first.first[0], 2) +
                          pow(this->state.p[1] - k.first.first[1], 2) +
                          pow(this->state.p[2] - k.first.first[2], 2) );

    // count the number nearby current pose
    if (delta_d <= this->keyframe_thresh_dist_ * 1.5){
      ++num_nearby;
    }

    // store into variable
    if (delta_d < closest_d) {
      closest_d = delta_d;
      closest_idx = keyframes_idx;
    }

    keyframes_idx++;

  }

  // INCREMENT 1b -- how OLD is that closest keyframe, and how far have I come
  // since I last laid one? Stock DLIO computes neither. `closest_idx` indexes
  // keyframes and keyframe_timestamps in lockstep (both are push_back'd together
  // under keyframes_mutex below), so the age is a lookup, not a search.
  double closest_age_s = 0.;
  if (closest_idx < (int)this->keyframe_timestamps.size()) {
    // Both sides are HEADER stamps. scan_stamp is the MEDIAN point time and
    // would introduce a systematic <= 50 ms offset against a stored header
    // stamp; on a 15 s threshold that is noise, but a replay harness comparing
    // this against exported poses should not have to know about it.
    closest_age_s = this->scan_header_stamp.toSec()
                  - this->keyframe_timestamps[closest_idx].toSec();
  }
  this->keyframe_age_last_s_ = closest_age_s;
  // Distance from MY OWN LAST keyframe, which on a revisit is a completely
  // different number from the distance to the closest keyframe of any age --
  // that difference IS the defect.
  double travel_since_last_kf = 0.;
  if (!this->keyframes.empty()) {
    const Eigen::Vector3f& lastp = this->keyframes.back().first.first;
    travel_since_last_kf = (this->state.p - lastp).norm();
  }

  // get closest pose and corresponding rotation
  Eigen::Vector3f closest_pose = this->keyframes[closest_idx].first.first;
  Eigen::Quaternionf closest_pose_r = this->keyframes[closest_idx].first.second;

  // calculate distance between current pose and closest pose from above
  float dd = sqrt( pow(this->state.p[0] - closest_pose[0], 2) +
                   pow(this->state.p[1] - closest_pose[1], 2) +
                   pow(this->state.p[2] - closest_pose[2], 2) );

  // calculate difference in orientation using SLERP
  Eigen::Quaternionf dq;

  if (this->state.q.dot(closest_pose_r) < 0.) {
    Eigen::Quaternionf lq = closest_pose_r;
    lq.w() *= -1.; lq.x() *= -1.; lq.y() *= -1.; lq.z() *= -1.;
    dq = this->state.q * lq.inverse();
  } else {
    dq = this->state.q * closest_pose_r.inverse();
  }

  double theta_rad = 2. * atan2(sqrt( pow(dq.x(), 2) + pow(dq.y(), 2) + pow(dq.z(), 2) ), dq.w());
  double theta_deg = theta_rad * (180.0/M_PI);

  // update keyframes
  bool newKeyframe = false;

  if (abs(dd) > this->keyframe_thresh_dist_ || abs(theta_deg) > this->keyframe_thresh_rot_) {
    newKeyframe = true;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_) {
    newKeyframe = false;
  }

  if (abs(dd) <= this->keyframe_thresh_dist_ && abs(theta_deg) > this->keyframe_thresh_rot_ && num_nearby <= 1) {
    newKeyframe = true;
  }

  // INCREMENT 1b -- THE AGE CLAUSE. Appended AFTER the three stock clauses and
  // never able to unset them, so with max_age_s = 0 (the default) this block is
  // dead and the decision above is stock DLIO's, unchanged.
  //
  // "The closest keyframe is older than max_age_s AND I have moved at least
  // min_travel_m since I laid my own last one." The second half is what keeps a
  // stationary rig from laying keyframes forever (this rig's bags carry 151
  // pauses); the first is what tells "I have been here" from "I have just been
  // here", which is the one question DLIO's Euclidean test cannot answer.
  if (this->keyframe_age_.on() && closest_age_s > this->keyframe_age_.max_age_s) {
    ++this->keyframe_age_stale_;
  }
  if (!newKeyframe &&
      dlio::repairs::age_clause(closest_age_s, travel_since_last_kf,
                                this->keyframe_age_, this->keyframe_age_travel_m_)) {
    newKeyframe = true;
    ++this->keyframe_age_fired_;
  }

  // SILENT NO-OP LAW. COMPUTED > 0 and APPLIED == 0 is a hard complaint: the
  // clause is armed, it has seen 500 scans whose closest keyframe was over the
  // age, and it has still never laid one -- which means min_travel_m is holding
  // it shut (a stationary rig, or a floor set larger than the drift), not that
  // the scene never went stale. 500 scans = 50 s of a 10 Hz sensor.
  if (this->keyframe_age_.on() && !this->keyframe_age_noop_reported_ &&
      this->keyframe_age_stale_ > 500 && this->keyframe_age_fired_ == 0) {
    this->keyframe_age_noop_reported_ = true;
    fprintf(stderr,
            "[REPAIR][ERROR] keyframe age clause armed (max_age_s=%.2f, "
            "min_travel_m=%.3f) and %ld scans saw a closest keyframe older than "
            "that, but NOT ONE keyframe was laid by it. The travel floor is "
            "refusing every one of them -- check dlio/odom/keyframe/min_travel_m "
            "against how far this rig actually moves between scans.\n",
            this->keyframe_age_.max_age_s, this->keyframe_age_travel_m_,
            (long)this->keyframe_age_stale_);
    fflush(stderr);
  }

  if (newKeyframe) {

    // update keyframe vector
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
    this->keyframes.push_back(std::make_pair(std::make_pair(this->lidarPose.p, this->lidarPose.q), this->current_scan));
    this->keyframe_timestamps.push_back(this->scan_header_stamp);
    this->keyframe_normals.push_back(this->gicp.getSourceCovariances());
    this->keyframe_transformations.push_back(this->T_corr);

    // --- E4: register the keyframe with the smoother. A keyframe IS a scan, so
    // there is no new variable: the smoother records which of its own keys this
    // keyframe's pose IS, and reads it back while that key is still inside the
    // window. The parallel vectors below are the map side of the same fact.
    {
      std::lock_guard<std::mutex> lk(this->kf_delta_mutex_);
      this->kf_pending_delta_.push_back(Eigen::Matrix4d::Identity());
      this->kf_has_delta_.push_back(0);
      Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();
      P0.block<3, 3>(0, 0) = this->lidarPose.q.toRotationMatrix().cast<double>();
      P0.block<3, 1>(0, 3) = this->lidarPose.p.cast<double>();
      this->kf_pose_now_.push_back(P0);
      this->kf_frozen_.push_back(0);
    }
    if (this->smoother_) {
      this->smoother_->noteKeyframe((int)this->keyframes.size() - 1);
    }
    lock.unlock();

  }

}

void dlio::OdomNode::setAdaptiveParams() {

  // Spaciousness
  float sp = this->metrics.spaciousness.back();

  if (sp < 0.5) { sp = 0.5; }
  if (sp > 5.0) { sp = 5.0; }

  this->keyframe_thresh_dist_ = sp;

  // Density
  float den = this->metrics.density.back();

  if (den < 0.5*this->gicp_max_corr_dist_) { den = 0.5*this->gicp_max_corr_dist_; }
  if (den > 2.0*this->gicp_max_corr_dist_) { den = 2.0*this->gicp_max_corr_dist_; }

  if (sp < 5.0) { den = 0.5*this->gicp_max_corr_dist_; };
  if (sp > 5.0) { den = 2.0*this->gicp_max_corr_dist_; };

  this->gicp.setMaxCorrespondenceDistance(den);

  // Concave hull alpha
  this->concave_hull.setAlpha(this->keyframe_thresh_dist_);

}

void dlio::OdomNode::pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames,
                                       std::size_t population) {

  // make sure dists is not empty
  if (!dists.size()) { return; }

  // INCREMENT 1a -- THE UNBOUNDED CANDIDATE LIST.
  //
  // Below, the max-heap holds at most k elements and `kth_element` is its top.
  // With fewer than k candidates that top is the LARGEST distance in the list,
  // so the inclusive test admits EVERY candidate, at ANY distance. For the knn
  // call that is right (the list is the whole keyframe population, and "the 20
  // nearest of the 3 that exist" is all 3). For the two HULL calls it is not:
  // the list is a filtered subset, there are nearer keyframes outside it, and
  // on ad3517ba the concave hull returns 4-8 vertices against kcc = 10 -- which
  // is how keyframes 0-3, sitting 68 m away in the opening chamber outside the
  // mapped corridor, are in every submap this bag builds.
  //
  // dlio/repairs.h carries the reasoning and the alternative (deriving the
  // hull's alpha) and why it was not taken.
  if (dlio::repairs::refuse_short_candidate_list(dists.size(), k, population)) {
    ++this->submap_short_refused_;
    return;
  }

  // maintain max heap of at most k elements
  std::priority_queue<float> pq;

  for (auto d : dists) {
    if (pq.size() >= k && pq.top() > d) {
      pq.push(d);
      pq.pop();
    } else if (pq.size() < k) {
      pq.push(d);
    }
  }

  // get the kth smallest element, which should be at the top of the heap
  float kth_element = pq.top();

  // get all elements smaller or equal to the kth smallest element
  for (int i = 0; i < dists.size(); ++i) {
    if (dists[i] <= kth_element)
      this->submap_kf_idx_curr.push_back(frames[i]);
  }

}

void dlio::OdomNode::buildSubmap(State vehicle_state) {

  // clear vector of keyframe indices to use for submap
  this->submap_kf_idx_curr.clear();

  // calculate distance between current pose and poses in keyframe set
  std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);
  std::vector<float> ds;
  std::vector<int> keyframe_nn;
  for (int i = 0; i < this->num_processed_keyframes; i++) {
    float d = sqrt( pow(vehicle_state.p[0] - this->keyframes[i].first.first[0], 2) +
                    pow(vehicle_state.p[1] - this->keyframes[i].first.first[1], 2) +
                    pow(vehicle_state.p[2] - this->keyframes[i].first.first[2], 2) );
    ds.push_back(d);
    keyframe_nn.push_back(i);
  }
  lock.unlock();

  // get indices for top K nearest neighbor keyframe poses
  // population == ds.size(): this list IS the whole keyframe population, so a
  // short list here is not the defect and stock behaviour is kept verbatim.
  this->pushSubmapIndices(ds, this->submap_knn_, keyframe_nn, ds.size());

  // get convex hull indices
  this->computeConvexHull();

  // get distances for each keyframe on convex hull
  std::vector<float> convex_ds;
  for (const auto& c : this->keyframe_convex) {
    convex_ds.push_back(ds[c]);
  }

  // get indices for top kNN for convex hull
  this->pushSubmapIndices(convex_ds, this->submap_kcv_, this->keyframe_convex, ds.size());

  // get concave hull indices
  this->computeConcaveHull();

  // get distances for each keyframe on concave hull
  std::vector<float> concave_ds;
  for (const auto& c : this->keyframe_concave) {
    concave_ds.push_back(ds[c]);
  }

  // get indices for top kNN for concave hull. `smkcc` on the [DEGEN] line is the
  // number of index entries THIS call pushed (before the union is de-duplicated):
  // the number the repair must drive to zero on a bag whose alpha-complex is
  // empty at alpha = threshD.
  const std::size_t before_kcc = this->submap_kf_idx_curr.size();
  this->pushSubmapIndices(concave_ds, this->submap_kcc_, this->keyframe_concave, ds.size());
  this->submap_kcc_added_ = (int)(this->submap_kf_idx_curr.size() - before_kcc);
  {   // the farthest keyframe THIS call let in -- the 68 m number, per scan
    double dk = 0.;
    for (std::size_t z = before_kcc; z < this->submap_kf_idx_curr.size(); ++z) {
      const int i = this->submap_kf_idx_curr[z];
      if (i >= 0 && i < (int)ds.size() && ds[i] > dk) { dk = ds[i]; }
    }
    this->submap_dkcc_ = dk;
  }

  // sort current and previous submap kf list of indices
  std::sort(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  std::sort(this->submap_kf_idx_prev.begin(), this->submap_kf_idx_prev.end());
  
  // remove duplicate indices
  auto last = std::unique(this->submap_kf_idx_curr.begin(), this->submap_kf_idx_curr.end());
  this->submap_kf_idx_curr.erase(last, this->submap_kf_idx_curr.end());

  // Submap COMPOSITION SIZE, after the union and the de-duplication: what the
  // registration is actually aiming at. 28 keyframes on RUN 2 at bag t 357.5,
  // 24 on the A/A, four of each 68 m away.
  this->submap_size_ = (int)this->submap_kf_idx_curr.size();
  {   // the farthest member of the submap actually built
    double dm = 0.;
    for (int i : this->submap_kf_idx_curr) {
      if (i >= 0 && i < (int)ds.size() && ds[i] > dm) { dm = ds[i]; }
    }
    this->submap_dmax_ = dm;
  }

  // SUBMAP.md section 6, fix (5): "plus a [SUBMAP][WARN] counting the refusals,
  // so the silent-no-op law binds in both directions". A run that never refuses
  // is a run whose hulls were always at least k long and the repair changed
  // nothing; a run that refuses on essentially every scan is telling you the
  // alpha-complex is empty and the kcc term has never done anything. Both are
  // findings; neither may be silent.
  if (!this->submap_refuse_reported_ && this->submap_short_refused_ > 1000) {
    this->submap_refuse_reported_ = true;
    fprintf(stderr,
            "[SUBMAP][WARN] %ld hull candidate lists shorter than k have been "
            "REFUSED (kcv=%d, kcc=%d). Stock DLIO would have admitted each whole "
            "list at unbounded distance. With alpha = threshD = %.3f m the "
            "concave hull's alpha-complex is empty on this scene, so the kcc "
            "term contributes nothing -- which is the true state of it, not a "
            "regression.\n",
            (long)this->submap_short_refused_, this->submap_kcv_, this->submap_kcc_,
            this->keyframe_thresh_dist_);
    fflush(stderr);
  }

  // check if submap has changed from previous iteration
  if (this->submap_kf_idx_curr != this->submap_kf_idx_prev){

    this->submap_hasChanged = true;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    // reinitialize submap cloud and normals
    pcl::PointCloud<PointType>::Ptr submap_cloud_ (boost::make_shared<pcl::PointCloud<PointType>>());
    std::shared_ptr<nano_gicp::CovarianceList> submap_normals_ (std::make_shared<nano_gicp::CovarianceList>());

    for (auto k : this->submap_kf_idx_curr) {

      // create current submap cloud
      lock.lock();
      *submap_cloud_ += *this->keyframes[k].second;
      lock.unlock();

      // grab corresponding submap cloud's normals
      submap_normals_->insert( std::end(*submap_normals_),
          std::begin(*(this->keyframe_normals[k])), std::end(*(this->keyframe_normals[k])) );
    }

    this->submap_cloud = submap_cloud_;
    this->submap_normals = submap_normals_;

    // Pause to prevent stealing resources from the main loop if it is running.
    this->pauseSubmapBuildIfNeeded();

    this->gicp_temp.setInputTarget(this->submap_cloud);
    this->submap_kdtree = this->gicp_temp.target_kdtree_;

    this->submap_kf_idx_prev = this->submap_kf_idx_curr;
  }
}

void dlio::OdomNode::buildKeyframesAndSubmap(State vehicle_state) {

  // --- E4: LAZY, and here on purpose. The smoother queues a world->world delta
  // per in-window keyframe on the lidar thread; it is applied HERE, on the
  // async submap thread, before the new keyframes are baked and before the
  // submap is composed -- so the kd-tree is rebuilt at most once per build and
  // never on the callback's critical path. Sub-floor motion never gets this
  // far: write_back() drops a delta below 5 mm / 0.05 deg.
  this->applyKeyframeDeltas();

  // transform the new keyframe(s) and associated covariance list(s)
    std::unique_lock<decltype(this->keyframes_mutex)> lock(this->keyframes_mutex);

  for (int i = this->num_processed_keyframes; i < this->keyframes.size(); i++) {
    pcl::PointCloud<PointType>::ConstPtr raw_keyframe = this->keyframes[i].second;
    std::shared_ptr<const nano_gicp::CovarianceList> raw_covariances = this->keyframe_normals[i];
    Eigen::Matrix4f T = this->keyframe_transformations[i];
    lock.unlock();

    Eigen::Matrix4d Td = T.cast<double>();

    pcl::PointCloud<PointType>::Ptr transformed_keyframe (boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud (*raw_keyframe, *transformed_keyframe, T);

    std::shared_ptr<nano_gicp::CovarianceList> transformed_covariances (std::make_shared<nano_gicp::CovarianceList>(raw_covariances->size()));
    std::transform(raw_covariances->begin(), raw_covariances->end(), transformed_covariances->begin(),
                   [&Td](Eigen::Matrix4d cov) { return Td * cov * Td.transpose(); });

    ++this->num_processed_keyframes;

    lock.lock();
    this->keyframes[i].second = transformed_keyframe;
    this->keyframe_normals[i] = transformed_covariances;

    this->publish_keyframe_thread = std::thread( &dlio::OdomNode::publishKeyframe, this, this->keyframes[i], this->keyframe_timestamps[i] );
    this->publish_keyframe_thread.detach();
  }

  lock.unlock();

  // Pause to prevent stealing resources from the main loop if it is running.
  this->pauseSubmapBuildIfNeeded();

  this->buildSubmap(vehicle_state);
}

void dlio::OdomNode::pauseSubmapBuildIfNeeded() {
  std::unique_lock<decltype(this->main_loop_running_mutex)> lock(this->main_loop_running_mutex);
  this->submap_build_cv.wait(lock, [this]{ return !this->main_loop_running; });
}

void dlio::OdomNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal.
  //
  // The banner is emitted field by field from a DETACHED thread spawned per scan,
  // so two banners can interleave with each other and a [DEGEN] flush from the
  // lidar thread can land inside a line. This banner is the instrument every
  // verdict on the Sandland bag was read from, so both printers serialize on
  // print_mutex_; the lock is held to the end of the function.
  std::lock_guard<std::mutex> print_lock(this->print_mutex_);

  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|               Direct LiDAR-Inertial Odometry v" << this->version_  << "               |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  // THE LINE ABOVE IS A MEAN OF 1/dt AND CANNOT SEE A DROPPED MESSAGE.
  // This one can: count over span, plus the worst interval and how many
  // intervals crossed dlio/odom/imu/gap_report_s. On the big Sandland bag the
  // line above read 632.44 Hz while this one would have read 471.95 -- the
  // 26 % the transport was losing (imu_delivery.h, IMU_CHECK.md section 5).
  {
    const long   rxn  = this->imu_rx_n_;
    const double rxhz = this->imu_rx_hz_;
    const double rxdt = this->imu_rx_maxdt_;
    const long   rxg  = this->imu_rx_gaps_;
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "IMU delivered: " + to_string_with_precision(rxhz, 2) + " Hz ("
         + std::to_string(rxn) + " msgs), dt max "
         + to_string_with_precision(rxdt * 1e3, 1) + " ms, gaps "
         + std::to_string(rxg)
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: keyframes: " + std::to_string(this->keyframes.size()) + ", "
                               + "deskewed points: " + std::to_string(this->deskew_size)
    << "|" << std::endl;

  // GICP degeneracy: the per-scan flag and the smallest eigenvalue of the
  // registration's translation information, so a run can be scored for
  // degenerate scans straight from `docker logs` — the same instrument every
  // other verdict on this bag came from. Always printed, so the line also
  // identifies which image is running.
  {
    // The box interior is 66 columns. This line is the only one here whose
    // length grows with the run (deg=11398/11398 ap=11398 rm=1234.5), so it is
    // hard-truncated: setw(66) only PADS, it never trims, and an over-long
    // string walks the right border out late in a long run — exactly when the
    // banner is being read most carefully.
    //
    // lmin was dropped in favour of the RATIO: the guard's own README explains
    // that the Hessian's absolute scale is meaningless under PLANE
    // regularization, so an eigenvalue on the banner invites the wrong
    // comparison. r is what the thresholds are applied to, w is what came out,
    // and rm is the cumulative metres of correction authority removed.
    char degbuf[160];
    if (!this->degen_params_.scoring()) {
      snprintf(degbuf, sizeof(degbuf), "Degeneracy  :: off");
    } else {
      snprintf(degbuf, sizeof(degbuf),
               "Degen(%s) :: F=%d r=%.4f w=%.2f d=%ld/%ld a=%ld rm=%.1f",
               this->degen_params_.enabled ? "guard" : "obs",
               (int)this->degen_flag_, (double)this->degen_ratio_min_,
               (double)this->degen_w_min_,
               (long)this->degen_degenerate_, (long)this->degen_scans_,
               (long)this->degen_applied_, (double)this->degen_removed_total_);
    }
    degbuf[66] = '\0';   // buffer is 160, so this only ever SHORTENS
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << degbuf << "|" << std::endl;
  }

  // INCREMENT 1 REPAIR RECORD. Always printed, so the line also identifies
  // which image is running and what it was armed with -- the three numbers a
  // reader needs to tell 1a from 1b from stock without opening the yaml.
  //   sm   submap composition size, keyframes, after the union
  //   kcc  index entries the CONCAVE-hull call pushed on that submap
  //   ref  candidate lists refused for being shorter than k (cumulative)
  //   age  max_age_s / keyframes laid by the age clause (cumulative)
  //   ab   the per-axis accel-bias clamp actually in force, m/s^2
  {
    char repbuf[192];
    if (this->keyframe_age_.on()) {
      snprintf(repbuf, sizeof(repbuf),
               "Repair :: sm=%d/%.0fm kcc=%d ref=%ld age=%.0fs/%ldkf rr=%.3f",
               (int)this->submap_size_, (double)this->submap_dmax_,
               (int)this->submap_kcc_added_,
               (long)this->submap_short_refused_, this->keyframe_age_.max_age_s,
               (long)this->keyframe_age_fired_,
               this->degen_rot_.valid ? this->degen_rot_.ratio_min() : -1.0);
    } else {
      snprintf(repbuf, sizeof(repbuf),
               "Repair :: sm=%d/%.0fm kcc=%d ref=%ld age=off rr=%.3f ab<=%.2f",
               (int)this->submap_size_, (double)this->submap_dmax_,
               (int)this->submap_kcc_added_,
               (long)this->submap_short_refused_,
               this->degen_rot_.valid ? this->degen_rot_.ratio_min() : -1.0,
               (double)this->geo_abias_clamp_[0]);
    }
    repbuf[66] = '\0';   // same hard truncation as the line above
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << repbuf << "|" << std::endl;
  }
  std::cout << "|                                                                   |" << std::endl;

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
