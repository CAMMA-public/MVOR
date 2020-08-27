#!/usr/bin/env bash
# '''
# Project: CAMMA-MVOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

CAMMA_MVOR="../"
# ground truth (--gt) and detection (--dt) json files are in standard coco format.
cd $CAMMA_MVOR/lib/
printf "\n\n"
printf "[Results for openpose]"
python3 eval_pck_2dpose.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/openpose_kps.json
printf "\n\n"

printf "[Results for openpose with multiscale testing]"
python3 eval_pck_2dpose.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/openpose_kps_multiscale.json
printf "\n\n"

printf "[Results for alphapose]"
python3 eval_pck_2dpose.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/alphapose_kps.json
printf "\n\n"

printf "[Results for rtpose]"
python3 eval_pck_2dpose.py --gt ../annotations/camma_mvor_2018.json --dt ../detections_results/rtpose_kps.json
printf "\n\n"