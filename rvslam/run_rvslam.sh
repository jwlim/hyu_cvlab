#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 & 

./rvslam_main -logtostderr 0 -egocentric \
        -calib 458.654,457.296,367.215,248.375,-0.283408,0.073959,0.0001935,0.00001,0 \
        -feature KLT_CV \
        -klt_max_features 200 -klt_redetect_thr 150 -klt_num_levels 4 \
        -feat_max_cnt 200 -feat_min_dist 10 -feat_f_threshold 1 \
        -image_files ~/Documents/Research/data/MH_01/cam0/image/IMG_%04d.png \
        -voc_file ../voctree_40_3.bin \
        -start 0 -step 1 -end 3900 -reduce_size 0 \
        -fast_th 20 -fast_window 5 -matching_th 5000 -matching_radius 100 \
        -overlap_th 0.90 -covisible_thr 30 -pose_inlier_thr 5 -num_keyframe 30 \
        -loop_th 30 \
        -ceres_num_iterations 30 -ceres_max_solver_time 0.30 -ceres_huber_loss_sigma 0.0056 \
        -pose_out pose2.txt -out side2.csio \

