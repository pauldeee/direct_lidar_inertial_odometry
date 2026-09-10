###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")

find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  pcl_ros
  message_generation
)

add_service_files(
  DIRECTORY srv
  FILES
  save_pcd.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    pcl_ros
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
    nano_gicp
    nanoflann
)

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS})

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)

# --- E4 / ARCHITECT A: the fixed-lag smoother ------------------------------
#
# GTSAM is REQUIRED, and deliberately so. A build that silently produced a
# dlio_odom_node with no smoother in it would be exactly the class of defect
# this lane exists to prevent: a recipe row that says `smoother/enabled: true`,
# a run that looks healthy, and no estimator behind it. Turning it off has to
# be TYPED (-DDLIO_WITH_SMOOTHER=OFF), never inherited from a missing package.
option(DLIO_WITH_SMOOTHER "Build the E4 fixed-lag smoother (needs GTSAM)" ON)
set(DLIO_SMOOTHER_SRC "")
set(DLIO_SMOOTHER_LIBS "")
if(DLIO_WITH_SMOOTHER)
  find_package(GTSAM QUIET)
  # BatchFixedLagSmoother lives in gtsam_unstable, which GTSAM exports as its
  # OWN cmake package (GTSAM_UNSTABLE), not as part of GTSAM's. A build that
  # found only GTSAM would compile every header and then fail at the link with
  # `cannot find -lgtsam_unstable`, so both are required by name here.
  # ...and it is a SIBLING cmake package directory, so a build that located
  # GTSAM by -DGTSAM_DIR alone would not find it. The hint is derived from
  # wherever GTSAM was actually found rather than hard-coded to /opt/gtsam.
  find_package(GTSAM_UNSTABLE QUIET HINTS "${GTSAM_DIR}/../GTSAM_UNSTABLE")
  if(NOT GTSAM_FOUND OR NOT GTSAM_UNSTABLE_FOUND)
    message(FATAL_ERROR
      "DLIO_WITH_SMOOTHER=ON but GTSAM (found=${GTSAM_FOUND}) and/or "
      "gtsam_unstable (found=${GTSAM_UNSTABLE_FOUND}) was not found. Install "
      "it -- the recipe image builds 4.2.2 into /opt/gtsam with "
      "GTSAM_BUILD_UNSTABLE=ON -- or say -DDLIO_WITH_SMOOTHER=OFF explicitly. "
      "It is not defaulted off on a missing package: see "
      "keeper/lio_concept/dlio_smoother/E4/IMPL/IMPL.md.")
  endif()
  # BatchFixedLagSmoother lives in gtsam_unstable in 4.2.x. A GTSAM built with
  # GTSAM_BUILD_UNSTABLE=OFF has the headers but not the library, and the link
  # is where that must fail -- not at runtime.
  message(STATUS "DLIO smoother: GTSAM ${GTSAM_VERSION} at ${GTSAM_INCLUDE_DIR}")
  include_directories(SYSTEM ${GTSAM_INCLUDE_DIR})
  add_definitions(-DDLIO_HAS_SMOOTHER=1)
  set(DLIO_SMOOTHER_SRC src/dlio/smoother.cc)
  # The IMPORTED targets, not the bare names: they carry the install prefix, so
  # a GTSAM outside the default link path links anyway and the failure mode
  # `cannot find -lgtsam_unstable` cannot come back.
  set(DLIO_SMOOTHER_LIBS gtsam gtsam_unstable)
  get_target_property(_gtsam_unstable_loc gtsam_unstable IMPORTED_LOCATION_RELEASE)
  message(STATUS "DLIO smoother: gtsam_unstable at ${_gtsam_unstable_loc}")
endif()

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc ${DLIO_SMOOTHER_SRC})
add_dependencies(dlio_odom_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(dlio_odom_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_odom_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp ${DLIO_SMOOTHER_LIBS})

# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
add_dependencies(dlio_map_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(dlio_map_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(dlio_map_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads)

# --- THE OFFLINE A/A HARNESS ------------------------------------------------
# Replays a recorded draw's own measurements through the SAME Smoother the node
# runs, so the in-DLIO estimator can be required to reproduce the offline refit
# before any node time is spent. It ships in the image on purpose: an A/A run
# against a DIFFERENT build of the same source proves nothing.
if(DLIO_WITH_SMOOTHER)
  add_executable(dlio_smoother_replay src/dlio/smoother_replay.cc src/dlio/smoother.cc)
  target_link_libraries(dlio_smoother_replay ${DLIO_SMOOTHER_LIBS} Threads::Threads)
endif()

# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} )
if(DLIO_WITH_SMOOTHER)
  install( TARGETS dlio_smoother_replay
           DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} )
endif()
install( DIRECTORY cfg launch
         DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )
