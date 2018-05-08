# RVSLAM
**Authors:**  : Euntae Hong(dragon1301@naver.com), Jongwoo Lim(jlim@hanyang.ac.kr)

RVSLAM is a open source library that estimates camera pose and reconstructs 3D environment in real time. The system finds the loop points using the vocabulary tree algorithm and performs pose graph optimization to resolve the accumulated error. How it works is listed below. This work was supported by Next-Generation Information Computing Development Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Science, ICT (NRF- 2017M3C4A7069369)

<a href="https://youtu.be/HNxaDOEJj_8" target="_blank"><img src='http://drive.google.com/uc?export=view&id=1Y1Os2O9k3eYeIVD67faLlLZrJWHotGQj' alt=“RVSLAM” border="10" /></a>

# 1. License
RVSLAM is released under a GPLv3 license. 
For a closed-source version of RVSLAM for commercial purposes, please contact the authors.

# 2. Prerequisites
We have tested the system under **MAC OS 10.11.6** and **Ubuntu 14.04**.


## For a list of all code/library dependencies
 - CMake 2.8.0
 - OpenCV 2.4.13
 - Eigen3 3.1.3
 - csio (Simple PLY Viewer) : https://github.com/jwlim/csio
 - libPNG
 - libJPEG
 - Ceres Solver 1.13
 - GoogleLib
  - GLOG 0.3.1
  - GFLAG 2.2.0
 - DC1394 (optional)

# 3. How to run RVSLAM 
We provide ‘run_rvslam.sh’ script with basic configure parameters.
If you want to use your own image or dataset, please set the parameters (ex. image path, camera calibration parameter, etc..).

 1. CSIO build
    - cd csio
    - mkdir build
    - cd build
    - cmake ..
    - make

 2. VIO build
    - mkdir build
    - cd build
    - cmake ..
    - make -j8

 3. run script
    - mkfifo side2.csio
    - sh run_rvslam.sh (run on build folder)

# 4. Run example
 1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets (ex. Machine Hall 01)
 2. Change image files name to the format ‘%04d.png’
 3. Set image path on ‘run_rvslam.sh’ script. The other parameters such as camera calibration parameters would be same as before.

# 5. Processing your own camera
 1. Set your own camera calibrated parameter on ‘run_rvslam.sh’ script.
 2. Add ‘-camera’ on ‘run_rvslam.sh’. This option will find your own camera using OpenCV VideoCapture api.