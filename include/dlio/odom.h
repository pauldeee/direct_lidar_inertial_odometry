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

#include "dlio/dlio.h"
#include "dlio/degeneracy.h"
#include "dlio/imu_delivery.h"
#include "dlio/repairs.h"
#include "dlio/smoother.h"
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>
#include <memory>

class dlio::OdomNode {

public:

  OdomNode(ros::NodeHandle node_handle);
  ~OdomNode();

  void start();

private:

  struct State;
  struct ImuMeas;

  void getParams();

  void callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc);
  void callbackImu(const sensor_msgs::Imu::ConstPtr& imu);

  void publishPose();  // per-scan corrected pose, stamped scan_stamp (MEDIAN)
  void publishHighRateInterval();  // dense IMU-rate poses for the just-finished scan interval

  void publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud);
  void publishKeyframe(std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                       pcl::PointCloud<PointType>::ConstPtr> kf, ros::Time timestamp);

  void getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc);
  void preprocessPoints();
  void deskewPointcloud();
  void initializeInputTarget();
  void setInputSource();

  void initializeDLIO();

  void getNextPose();
  void scoreDegeneracy();   // score this scan's GICP geometry; see dlio/degeneracy.h
  void logDegeneracy();     // report it AFTER updateState, so `removed` is THIS scan's
  // --- E4 / ARCHITECT A: the nudge. See dlio/smoother.h for the whole design.
  void smootherUpdate();       // solve, then the SIX writes, at odom.cc's :1256
  void logSmoother();          // the [SMOOTH] record, one line per scan
  void applyKeyframeDeltas();  // in buildKeyframesAndSubmap, BEFORE buildSubmap
  bool imuMeasFromTimeRange(double start_time, double end_time,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                            boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                 const std::vector<double>& sorted_timestamps);
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
    integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                         const std::vector<double>& sorted_timestamps,
                         boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                         boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it);
  void propagateGICP();

  void propagateState(const ImuMeas& m);
  void propagateStateScanWindow();
  void updateState();

  void setAdaptiveParams();
  void setKeyframeCloud();

  void computeMetrics();
  void computeSpaciousness();
  void computeDensity();

  sensor_msgs::Imu::Ptr transformImu(const sensor_msgs::Imu::ConstPtr& imu);

  void updateKeyframes();
  void computeConvexHull();
  void computeConcaveHull();
  // `population` is the number of keyframes `dists` was computed over. It is
  // what separates "the k nearest of a population smaller than k" (correct,
  // and the only submap there is at run start) from "the whole of a hull
  // shorter than k, at any distance" (the defect: four keyframes 68 m away
  // in every submap on ad3517ba). See dlio/repairs.h.
  void pushSubmapIndices(std::vector<float> dists, int k, std::vector<int> frames,
                         std::size_t population);
  void buildSubmap(State vehicle_state);
  void buildKeyframesAndSubmap(State vehicle_state);
  void pauseSubmapBuildIfNeeded();

  void debug();

  ros::NodeHandle nh;
  ros::Timer publish_timer;

  // Subscribers
  ros::Subscriber lidar_sub;
  ros::Subscriber imu_sub;

  // Dedicated single-threaded callback queue + spinner for the lidar subscriber.
  // callbackPointCloud is NOT re-entrant (it mutates shared scan/keyframe state),
  // so a DEEP pointcloud queue spun by the node's multi-threaded AsyncSpinner(0)
  // would run it CONCURRENTLY under lag and corrupt the buffers ("Low number of
  // points"). Pinning the lidar callback to its own 1-thread spinner serializes
  // it while the deep queue buffers the backlog -> DLIO flushes every scan at its
  // own pace, deterministically, regardless of playback speed. IMU and the rest
  // stay on the global queue/spinner, so IMU is never starved by lidar work.
  ros::CallbackQueue lidar_cb_queue;
  std::shared_ptr<ros::AsyncSpinner> lidar_spinner;

  // Publishers
  ros::Publisher odom_pub;
  ros::Publisher pose_pub;
  ros::Publisher path_pub;
  ros::Publisher kf_pose_pub;
  ros::Publisher kf_cloud_pub;
  ros::Publisher deskewed_pub;

  // ROS Msgs
  nav_msgs::Odometry odom_ros;
  geometry_msgs::PoseStamped pose_ros;
  nav_msgs::Path path_ros;
  geometry_msgs::PoseArray kf_pose_ros;

  // Flags
  std::atomic<bool> dlio_initialized;
  std::atomic<bool> first_valid_scan;
  std::atomic<bool> first_imu_received;
  std::atomic<bool> imu_calibrated;
  std::atomic<bool> submap_hasChanged;
  std::atomic<bool> gicp_hasConverged;
  std::atomic<bool> deskew_status;
  std::atomic<int> deskew_size;

  // Threads
  std::thread publish_thread;
  std::thread publish_keyframe_thread;
  std::thread metrics_thread;
  std::thread debug_thread;

  // Trajectory
  std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory;
  double length_traversed;

  // Keyframes
  std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>,
                        pcl::PointCloud<PointType>::ConstPtr>> keyframes;
  std::vector<ros::Time> keyframe_timestamps;
  std::vector<std::shared_ptr<const nano_gicp::CovarianceList>> keyframe_normals;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> keyframe_transformations;
  std::mutex keyframes_mutex;
  // publishKeyframe() runs on a DETACHED thread, one per unprocessed keyframe,
  // and mutates kf_pose_ros before publishing it. roscpp serializes a message in
  // two passes (measure, then write into an exactly-sized buffer); a concurrent
  // push_back between the passes overruns the buffer and throws
  // StreamOverrunException out of a thread with no handler => std::terminate.
  std::mutex kf_pose_mutex;

  // Sensor Type
  dlio::SensorType sensor;

  // Frames
  std::string odom_frame;
  std::string baselink_frame;
  std::string lidar_frame;
  std::string imu_frame;

  // Preprocessing
  pcl::CropBox<PointType> crop;
  pcl::VoxelGrid<PointType> voxel;

  // Point Clouds
  pcl::PointCloud<PointType>::ConstPtr original_scan;
  pcl::PointCloud<PointType>::ConstPtr deskewed_scan;
  pcl::PointCloud<PointType>::ConstPtr current_scan;

  // Keyframes
  pcl::PointCloud<PointType>::ConstPtr keyframe_cloud;
  int num_processed_keyframes;

  pcl::ConvexHull<PointType> convex_hull;
  pcl::ConcaveHull<PointType> concave_hull;
  std::vector<int> keyframe_convex;
  std::vector<int> keyframe_concave;

  // Submap
  pcl::PointCloud<PointType>::ConstPtr submap_cloud;
  std::shared_ptr<const nano_gicp::CovarianceList> submap_normals;
  std::shared_ptr<const nanoflann::KdTreeFLANN<PointType>> submap_kdtree;

  std::vector<int> submap_kf_idx_curr;
  std::vector<int> submap_kf_idx_prev;

  bool new_submap_is_ready;
  std::future<void> submap_future;
  std::condition_variable submap_build_cv;
  bool main_loop_running;
  std::mutex main_loop_running_mutex;

  // Timestamps
  ros::Time scan_header_stamp;
  double scan_stamp;
  double prev_scan_stamp;
  double scan_dt;
  std::vector<double> comp_times;
  std::vector<double> imu_rates;
  std::vector<double> lidar_rates;

  double first_scan_stamp;
  double elapsed_time;

  // GICP
  nano_gicp::NanoGICP<PointType, PointType> gicp;
  nano_gicp::NanoGICP<PointType, PointType> gicp_temp;

  // Transformations
  Eigen::Matrix4f T, T_prior, T_corr;
  Eigen::Quaternionf q_final;

  Eigen::Vector3f origin;

  struct Extrinsics {
    struct SE3 {
      Eigen::Vector3f t;
      Eigen::Matrix3f R;
    };
    SE3 baselink2imu;
    SE3 baselink2lidar;
    Eigen::Matrix4f baselink2imu_T;
    Eigen::Matrix4f baselink2lidar_T;
  }; Extrinsics extrinsics;

  // IMU
  ros::Time imu_stamp;
  double first_imu_stamp;
  double prev_imu_stamp;
  double imu_dp, imu_dq_deg;

  struct ImuMeas {
    double stamp;
    double dt; // defined as the difference between the current and the previous measurement
    Eigen::Vector3f ang_vel;
    Eigen::Vector3f lin_accel;
  }; ImuMeas imu_meas;

  boost::circular_buffer<ImuMeas> imu_buffer;
  std::mutex mtx_imu;
  std::condition_variable cv_imu_stamp;

  // --- E4: THE RAW IMU TAP ---------------------------------------------------
  // `imu_buffer` above is BIAS-CORRECTED AT ARRIVAL with whatever bias was
  // current (callbackImu), so it cannot be reused by an estimator that
  // re-estimates bias; and it has been through transformImu(), which rotates
  // into baselink and adds a lever-arm term built from a FINITE DIFFERENCE of
  // omega at 640 Hz against a gyro whose own rms is 0.264 deg/s. gtsam's
  // body_P_sensor does that job analytically and without differentiating noise.
  // So the smoother gets its own tap: the RAW sensor-frame sample, taken at the
  // top of callbackImu before anything touches it.
  struct RawImu {
    double stamp;
    Eigen::Vector3d accel;   // m/s^2, SENSOR frame, no bias removed
    Eigen::Vector3d gyro;    // rad/s, SENSOR frame, no bias removed
  };
  std::deque<RawImu> raw_imu_buffer_;
  std::mutex mtx_raw_imu_;
  double raw_imu_keep_s_;      // how far back the tap is kept (lag + margin)
  std::atomic<long> raw_imu_n_;

  static bool comparatorImu(ImuMeas m1, ImuMeas m2) {
    return (m1.stamp < m2.stamp);
  };

  // Geometric Observer
  struct Geo {
    bool first_opt_done;
    std::mutex mtx;
    double dp;
    double dq_deg;
    Eigen::Vector3f prev_p;
    Eigen::Quaternionf prev_q;
    Eigen::Vector3f prev_vel;
  }; Geo geo;

  // State Vector
  struct ImuBias {
    Eigen::Vector3f gyro;
    Eigen::Vector3f accel;
  };

  struct Frames {
    Eigen::Vector3f b;
    Eigen::Vector3f w;
  };

  struct Velocity {
    Frames lin;
    Frames ang;
  };

  struct State {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
    Velocity v;
    ImuBias b; // imu biases in body frame
  }; State state;

  struct Pose {
    Eigen::Vector3f p; // position in world frame
    Eigen::Quaternionf q; // orientation in world frame
  };
  Pose lidarPose;
  Pose imuPose;

  // Metrics
  struct Metrics {
    std::vector<float> spaciousness;
    std::vector<float> density;
  }; Metrics metrics;

  std::string cpu_type;
  std::vector<double> cpu_percents;
  clock_t lastCPU, lastSysCPU, lastUserCPU;
  int numProcessors;

  // Parameters
  std::string version_;
  int num_threads_;
  bool verbose;

  bool deskew_;

  double gravity_;

  bool time_offset_;

  bool adaptive_params_;

  // Opt-in (~dlio/deterministic): force single-threaded GICP so its OpenMP
  // reduction order is fixed. Cuts run-to-run pose noise ~7x (median ~11mm ->
  // ~1.5mm) but is NOT bit-identical: nano_gicp covariance computation and the
  // async submap-build timing still vary. Off by default (multi-threaded,
  // faster). NOTE: playback-rate independence does NOT need this -- the cross-
  // rate diff already equals the same-rate diff. Use only if you want maximum
  // run-to-run reproducibility and can accept the slowdown (the deep queue
  // absorbs the extra lag for offline recording).
  bool deterministic_;

  // High-rate odom (~dlio/highrate_odom). Replaces the removed 100Hz wall-clock
  // timer with a DATA-DRIVEN emitter: at the end of each processed scan we IMU-
  // integrate the just-finished scan interval (prev corrected pose -> this scan)
  // and publish dense (~highrate_odom_hz) poses on the odom topic, stamped at the
  // IMU times. It runs in the scan callback so it flushes at DLIO's processing
  // rate (no wall-clock => no lag-warp), is bounded per-interval (no runaway
  // dead-reckoning under backlog), and is read-only (never feeds the SLAM state,
  // so no divergence). Off by default; the recorder launch sets it on.
  bool publish_highrate_odom_;
  double highrate_odom_dt_;            // decimation period (s); 0.01 = 100Hz
  bool highrate_anchor_valid_;
  double highrate_prev_stamp_;
  Eigen::Vector3f highrate_prev_p_;
  Eigen::Quaternionf highrate_prev_q_;
  Eigen::Vector3f highrate_prev_v_;

  double obs_submap_thresh_;
  double obs_keyframe_thresh_;
  double obs_keyframe_lag_;

  double keyframe_thresh_dist_;
  double keyframe_thresh_rot_;

  int submap_knn_;
  int submap_kcv_;
  int submap_kcc_;
  double submap_concave_alpha_;

  // --- INCREMENT 1 repairs (slamlab) -----------------------------------------
  // (1a) the short-candidate-list refusal. `smkcc` is how many keyframes the
  // CONCAVE-hull call admitted on the scan just built -- the number that must go
  // to zero on this bag -- and `refused` counts every list the guard turned away
  // over the run, so the repair cannot be a silent no-op in either direction.
  std::atomic<long> submap_short_refused_;   // lists refused (cumulative)
  std::atomic<int>  submap_kcc_added_;       // keyframes admitted by kcc, THIS submap
  std::atomic<int>  submap_size_;            // keyframes in the submap, THIS submap
  // The distances that say whether the 68 m keyframes are actually gone. dmax
  // is the farthest member of the submap the run built; dkcc is the farthest
  // one the CONCAVE-hull call admitted (0 when it admitted none).
  std::atomic<double> submap_dmax_;
  std::atomic<double> submap_dkcc_;
  bool submap_refuse_reported_;              // one-shot: the hull is always short

  // (1b) the keyframe AGE clause. OFF unless max_age_s > 0.
  dlio::repairs::KeyframeAge keyframe_age_;
  double keyframe_age_travel_m_;             // resolved 0.25*threshD unless set
  std::atomic<long> keyframe_age_fired_;     // keyframes laid BY the age clause
  std::atomic<long> keyframe_age_stale_;     // scans whose closest kf was older than max_age_s
  std::atomic<double> keyframe_age_last_s_;  // age of the closest kf, last scan
  bool keyframe_age_noop_reported_;          // one-shot: armed, stale seen, never fired

  // ROTATION-BLOCK INSTRUMENT (dlio/degeneracy.h rotation_record). Read-only:
  // no action is derived from it anywhere. Every lambda/r/w published by this
  // programme so far is TRANSLATION, while the divergence onset is a HEADING
  // error -- this is the column that could see that, fitted on raw observe data
  // before anyone proposes an action on it.
  dlio::degeneracy::RotRecord degen_rot_;

  // E2 -- THE TRUE 6x6 (dlio/degeneracy.h hessian_record). Read-only, like the
  // rotation record above and for the same reason: E0 fitted its attitude
  // weight on rlam, which is anchored at the WORLD ORIGIN, and the fit came out
  // 6.62x apart between the big bag and Dave against a 6.85 ratio of squared
  // distances to that origin. The cross block cancels that lever arm exactly
  // and no summary of the two diagonal blocks can (E2.md). Nothing acts on it.
  dlio::degeneracy::HessianRecord degen_h_;
  // The GICP pose of the PREVIOUS scored scan, so the record can carry the
  // registration's own relative pose -- the measurement a BetweenFactor wants -
  // instead of only |dp|. Taken at scoreDegeneracy(), i.e. BEFORE the optional
  // guard_pose adjustment; on every arm this lane runs guard_pose is false and
  // the two coincide, which is stated rather than assumed.
  Eigen::Matrix4f degen_T_prev_;
  bool            degen_T_prev_valid_;
  Eigen::Matrix<double, 6, 1> degen_dp6_;    // rel. pose prev->this, [rotvec ; t]
  bool            degen_dp6_valid_;         // false on the first scored scan only
  Eigen::Matrix<double, 6, 1> degen_corr6_;  // T_prior -> T, the registration's own correction
  Eigen::Matrix<double, 6, 1> degen_innov6_; // state -> lidarPose, the observer innovation
  std::atomic<long> degen_h6_written_;       // lines actually carrying a valid 6x6 (APPLIED)
  bool degen_h6_noop_reported_;              // one-shot: scored for a whole run, never dumped

  // (N57) the accel-bias clamp derived from THIS run's 3 s init calibration.
  double geo_abias_margin_;                  // 0 = off = the constant below
  Eigen::Vector3f geo_abias_clamp_;          // what updateState() actually uses
  bool geo_abias_derived_;                   // true once the calibration landed

  bool densemap_filtered_;
  // When true (default), the PUBLISHED/saved deskewed cloud is the full-density
  // (non-voxelized) deskewed_scan instead of the voxelized current_scan. GICP,
  // keyframes and the map keep using the voxelized current_scan — this only
  // affects the published `deskewed` topic (and thus the exporter's saved PCDs).
  bool dense_output_;
  bool wait_until_move_;

  double crop_size_;

  bool vf_use_;
  double vf_res_;

  bool imu_calibrate_;
  bool calibrate_gyro_;
  bool calibrate_accel_;
  bool gravity_align_;
  double imu_calib_time_;
  int imu_buffer_size_;
  // ROS subscriber queue depths. Default 1/1000 = live (drop stale scans to stay
  // real-time). For deterministic bag recording set large so no scan/IMU is ever
  // dropped under processing lag -- the output then depends only on the bag, not
  // on playback speed. Paired with a large imu_buffer_size_ so late scans still
  // find their IMU window.
  int sub_pointcloud_queue_;
  int sub_imu_queue_;
  Eigen::Matrix3f imu_accel_sm_;

  int gicp_min_num_points_;
  int gicp_k_correspondences_;
  double gicp_max_corr_dist_;
  int gicp_max_iter_;
  double gicp_transformation_ep_;
  double gicp_rotation_ep_;
  double gicp_init_lambda_factor_;

  double geo_Kp_;
  double geo_Kv_;
  double geo_Kq_;
  double geo_Kab_;
  double geo_Kgb_;
  double geo_abias_max_;
  double geo_gbias_max_;

  // --- GICP degeneracy guard (slamlab) ---------------------------------------
  dlio::degeneracy::Params  degen_params_;
  dlio::degeneracy::Weights degen_w_;        // this scan's verdict; lidar-callback thread only

  // Banner/telemetry. Written by the lidar callback, read by the debug thread —
  // atomic so the banner never tears, same idiom as deskew_size.
  std::atomic<bool>   degen_flag_;           // this scan was scored degenerate
  std::atomic<double> degen_ratio_min_;      // lambda_min / lambda_max this scan
  std::atomic<double> degen_w_min_;          // smallest observability weight this scan
  std::atomic<double> degen_removed_;        // |err| removed from the observer THIS scan (metres)
  // INSTRUMENT ONLY, no arithmetic depends on either: the two numbers that did
  // not exist in any artifact of the first three runs and had to be inferred
  // from pose steps (aa/AA_ANALYSIS.md section 8). degen_innov_ is the RAW
  // observer innovation |pin - state.p| BEFORE the guard touches it - the
  // quantity innov_max_m would gate, so arm (c) cannot be calibrated without
  // it. degen_dp_ is the GICP translation increment |T.p - T_prior.p| of the
  // same scan: registration movement, as distinct from observer disagreement.
  std::atomic<double> degen_innov_;          // |err_raw| this scan (metres)
  std::atomic<double> degen_dp_;             // |T.p - T_prior.p| this scan (metres)
  // Cumulative |err_raw - err|, metres, over the whole run. degen_removed_ is
  // per-scan and degen_applied_ counts SCANS, so neither can answer the one
  // question a finished run has to answer: HOW MUCH correction authority did the
  // guard take away in total? One atomic add per scan buys that number.
  std::atomic<double> degen_removed_total_;
  std::atomic<long>   degen_scans_;          // scans scored
  std::atomic<long>   degen_degenerate_;     // scans with w_min < 1          (COMPUTED)
  std::atomic<long>   degen_applied_;        // scans where something was removed (APPLIED)
  std::atomic<long>   degen_invalid_;        // scans refused (stale Hessian / too few corr.)
  bool degen_noop_reported_;                 // one-shot silent-no-op complaint (armed, nothing applied)
  bool degen_blind_reported_;                // one-shot complaint: armed, nothing ever SCORED degenerate

  // --- E4 / ARCHITECT A -------------------------------------------------------
  // The smoother, its ledger, and the plumbing that carries a smoothed keyframe
  // pose from the lidar thread into the async submap builder. `enabled` is
  // FALSE in the image: the recipe row turns it on, so the product path is
  // untouched by the presence of this code.
  dlio::smoother::Params smoother_params_;
  std::unique_ptr<dlio::smoother::Smoother> smoother_;
  dlio::smoother::Solution smoother_sol_;
  dlio::smoother::WriteBackReport smoother_rep_;
  dlio::smoother::NoOpLedger smoother_ledger_;
  bool smoother_bias_seeded_;
  bool smoother_noop_reported_;
  long smoother_scans_;                      // scans handed to the smoother
  std::vector<double> smoother_solve_ms_;    // the compute-budget series
  double smoother_solve_ms_max_;

  // KEYFRAME WRITE-BACK. The lidar thread queues a world->world delta per
  // in-window keyframe; buildKeyframesAndSubmap() applies it to the cloud, the
  // covariances and the stored transform BEFORE composing the submap, and the
  // kd-tree is rebuilt lazily on the next align because the composition changed.
  // FREEZE ON MARGINALISATION: once a keyframe's scan leaves the window it is
  // frozen forever and queueDelta() REFUSES it -- counted, never silent.
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> kf_pending_delta_;
  std::vector<char> kf_has_delta_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> kf_pose_now_;
  std::vector<char> kf_frozen_;
  std::mutex kf_delta_mutex_;
  std::atomic<long> smoother_kf_applied_;
  std::atomic<long> smoother_kf_refused_;
  std::atomic<bool> smoother_submap_dirty_;

  // The sink is a nested type so it can reach the vectors above without making
  // any of them public. It holds a bare pointer to the node, which outlives it.
  struct KfSink : public dlio::smoother::KeyframeSink {
    explicit KfSink(dlio::OdomNode* n) : node(n) {}
    bool queueDelta(int kf_index, const Eigen::Matrix4d& delta,
                    const Eigen::Matrix4d& pose_new) override;
    void markSubmapDirty() override;
    bool poseOf(int kf_index, Eigen::Matrix4d* out) const override;
    dlio::OdomNode* node;
  };
  std::unique_ptr<KfSink> kf_sink_;

  // stdout is one buffer. logDegeneracy() printf+fflush runs on the lidar
  // callback thread while debug() writes the status banner field by field with
  // std::cout from a DETACHED thread per scan; with sync_with_stdio they share
  // stdio's buffer, so a [DEGEN] flush can split a banner line in two — and that
  // banner is the instrument every verdict on the Sandland bag was read from.
  // Both printers take this lock, and so do two overlapping banners.
  std::mutex print_mutex_;

  // DELIVERED IMU — the transport witness this node never had.
  //
  // `imu_rates` + the banner's mean(1/dt) cannot see a dropped message
  // (imu_delivery.h says why, with the numbers). These four say how much of the
  // stream actually arrived: count over span, the worst interval, and how many
  // intervals crossed imu_gap_s_. Written only by callbackImu, which roscpp
  // serializes per subscription; mirrored into atomics because the status
  // banner and the [DEGEN] line read them from other threads.
  dlio::imu_delivery::Counter imu_rx_;
  double imu_gap_s_;                         // what counts as a gap (seconds)
  bool   imu_gap_reported_;                  // one-shot "this stream is lossy"
  std::atomic<long>   imu_rx_n_;
  std::atomic<double> imu_rx_hz_;
  std::atomic<double> imu_rx_maxdt_;
  std::atomic<long>   imu_rx_gaps_;

};
