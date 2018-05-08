#!/bin/bash
../csio/build/csio_glviewer -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 & 
sudo ../csio/build/csiomod_imu_1394 --modestr=fmt7.0.mono8.640.480.320.240. --port=ttyUSB1 | \

#./rvslam_live save.txt -alsologtostderr -egocentric \
./rvslam_live -egocentric \
        -calib 687,688,343.01463,227.58580,0,0 \
        -klt_max_features 10000 -klt_redetect_thr 150 -klt_num_levels 4 \
        -image_files ../../rvslam/data/lab/fast/capture_image_%04d.png -start 0 -step 1 -end 50000 -reduce_size 0 \
        -fast_th 30 -matching_th 5000 -matching_radius 20 -overlap_th 0.95 -covisible_thr 50 -ceres_huber_loss_sigma 0.0056 \
        -pose_inlier_thr 4 -num_keyframes 10 -ceres_num_iterations 50 \
        -loop_th 50 \
        -pose_out pose2.txt -out side2.csio \

#        -out - | ../csio/build/csio_glviewer -alsologtostderr -in - -pause -grid 100,5,-1 -cam 0,1,-3.5,0.2618,0,0 -fov 45 #-cap dump/%04d.png

#        -show_all_keyframes -out top2.csio
#        -pose_out pose1.txt -out tmp1.csio \

# Input sequences
#        -image ../data/130613_1206/rectified_image/left/%06d.png -depthmap ../data/130613_1206/depth_image/half/%05d.png -start 500 -end 3115 \
#        -image ../data/130613_1443/rectified_image/left/%06d.png -depthmap ../data/130613_1443/depth_image/half/%05d.png -start 1 -end 1513 \

# Viewers
#        -out - | ../csio/build/csio_glviewer -alsologtostderr -in ./side2.csio -pause -grid 100,5,-1 -cam 0,5,-5,0.698132,0,0 #-cap dump/%04d.png
#        -out - | ../csio/build/csio_glviewer -alsologtostderr -in - -pause -cap dump/%04d.png


#        -pose_inlier_thr 0.5 -keyframe_translation_thr 1000 \
#        -stereo_calib 707.0912,707.0912,601.8873,183.1104,-379.8145 \
#        -image kitti/04/image_0/%06d.png -depthmap lge_depthmap/04/dispimg_%06d.png -start 0 -end 272 \
