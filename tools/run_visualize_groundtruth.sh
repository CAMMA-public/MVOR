#!/usr/bin/env bash
# '''
# Project: CAMMA-MVOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
# define the path to the ground truth input json
inp_json="../annotations/camma_mvor_2018.json"

# root directory to the MVOR blurred dataset
img_dir="../dataset"

save_gt_path="../results_gt"

# execute the visualization script (press 'q' to quit; any key to continue)
PYTHONPATH=../ python3 -m lib.visualize_groundtruth \
       --inp_json ${inp_json} \
       --img_dir ${img_dir} \
       --show_ann true \
       --viz_2D true \
       --show_3dto2dproj true \
       --viz_3D true \
       --show_pose_variability false

# **********************  parameter explanation ***********************************************
# --inp_json => path to input json file containing 2D and 3D keypoints.
# --img_dir => path to the image directory.
# --show-ann => if false, will show only the dataset images without any annotations.
# --viz_2D => if true, will show the 2D keypoint annotations.
# --show_3dto2dproj => if true, will show the projections of 3D keypoints on all the frames.
# --viz_3D => if true, will show the 3D keypoints.
# --save_gt => if true, it will not render anything on the screen but write the visualization on the given path.
# --show_pose_variability = if true, it will show the 2D annotations pose variability of the MVOR dataset.
# *********************************************************************************************