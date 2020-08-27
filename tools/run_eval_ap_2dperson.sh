#!/usr/bin/env bash
# '''
# Project: CAMMA-MVOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

CAMMA_MVOR="../"
# ground truth (--gt) and detection (--dt) json files are in standard coco format.
cd $CAMMA_MVOR/lib/

printf "[Results for faster rcnn]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/faster_rcnn_bbox.json
printf "\n\n"

printf "[Results for deformable conv-nets R-FCN ]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/dfcnet_rfcn_bbox.json
printf "\n\n"

printf "[Results for openpose default ]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/openpose_bbox.json
printf "\n\n"

printf "[Results for openpose multiscale ]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/openpose_bbox_multiscale.json
printf "\n\n"

printf "[Results for alphapose ]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/alphapose_bbox.json
printf "\n\n"

printf "[Results for rtpose ]\n"
python3 eval_ap_2dperson.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/rtpose_bbox.json
printf "\n\n"

