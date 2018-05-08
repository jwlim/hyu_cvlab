#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 & 

./rvslam_main -logtostderr 0 -egocentric \
  -calib 458.654,457.296,367.215,248.375,-0.283408,0.073959,0.0001935,0.00001,0 -feature KLT \
  -klt_max_features 300 -klt_redetect_thr 250 -klt_num_levels 4 -klt_min_cornerness 10 \
  -keyframe_translation_thr 1 -check_keyframe_th 250 -reduce_size 0 \
  -fast_th 20 -fast_window 5 -matching_th 100 -matching_radius 100 -overlap_th 0.75 -covisible_thr 30 -ceres_huber_loss_sigma 0.0056 \
  -pose_inlier_thr 3 -num_keyframe 20 -num_keyframe_lba 20 -num_keyframe_vio 20 -num_active_keyframe 5 -ceres_num_iterations 300 \
  -ceres_huber_loss_velocity 3 -ceres_huber_loss_accel 10 -ceres_huber_loss_gyro 0.01 \
  -loop_th 30 \
  -out side2.csio \
  -root ~/Documents/icra2016/okvis/data/mav3/ \
  -start 100 -step 1 -end 2020 -is_mav -display_cam_size 0.8 \
  -show_all_keyframes -show_reprojected_pts \
  -imu_scale_accel 0.1 -imu_scale_gyro 0.1 -imu_scale_accel_batch 0.1 -velocity_scale 0.1 \
  -accel_weight 1 -gyro_weight 0.1 -ceres_visual_weight 1 -dump \
  -edge_th 0.9 -dense_recon \
  -searching_sigma 0.1 \

  #-calib 720,720,320,240,0,0,0,0 -feature KLT \
