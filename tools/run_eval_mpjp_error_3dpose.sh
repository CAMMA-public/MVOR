#!/usr/bin/env bash
# '''
# Project: CAMMA-MVOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''

CAMMA_MVOR="../"
cd $CAMMA_MVOR/lib/

printf "[                                                  mv3dreg MPJP error results on the blur images                                                                     ]"
matlab -nodisplay -nojvm -nosplash -nodesktop -r "eval_mpjp_error_3dpose('../detections_results/mv3dreg.mat');exit;"
printf "\n\n"