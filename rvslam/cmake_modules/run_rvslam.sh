#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 & 

./rvslam_main -logtostderr 0 -egocentric \
        -calib 687,688,343.01463,227.58580,0,0 -feature FAST \
        -klt_max_features 300 -klt_redetect_thr 150 -klt_num_levels 4 \
        -image_files ../data/lab/seq5/capture_image_%04d.png -start 10 -step 1 -end 2000 -reduce_size 0 \
        -fast_th 20 -fast_window 5 -matching_th 5000 -matching_radius 100 -overlap_th 0.90 -covisible_thr 30 -ceres_huber_loss_sigma 0.0056 \
        -pose_inlier_thr 4 -num_keyframes 7 -ceres_num_iterations 50 \
        -keyframe_translation_thr 1 \
        -loop_th 30 \
        -pose_out pose2.txt -out side2.csio \

