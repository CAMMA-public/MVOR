'''
File: eval_ap_2dperson.py
Project: CAMMA-MVOR
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import sys
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Compute the AP for the the person detection results')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='path to input json file containing bounding boxes'
    )
    parser.add_argument(
        '--dt',
        type=str,
        default='true',
        help='path to detection json file containing bounding boxes'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    gt = args.gt
    dt = args.dt
    annType = 'bbox'

    cocoGt = COCO(gt)

    for i, ann in enumerate(cocoGt.anns):
        cocoGt.anns[ann]['id'] = i
        if not 'iscrowd' in cocoGt.anns[ann].keys():
            cocoGt.anns[ann]['iscrowd'] = 0
        if not 'area' in cocoGt.anns[ann].keys():
            cocoGt.anns[ann]['area'] = cocoGt.anns[ann]['bbox'][2] * cocoGt.anns[ann]['bbox'][3]

    # initialize COCO detections api
    cocoDt = cocoGt.loadRes(dt)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    rec = cocoEval.eval['recall']
    np.set_printoptions(precision=3)
    print('Recall for IoU = 0.5, All Areas, All categories, max_det = 1',rec[0,:,0,0])
    print('Recall for IoU = 0.5, All Areas, All categories, max_det = 10',rec[0,:,0,1])
    print('Recall for IoU = 0.5, All Areas, All categories, max_det = 100',rec[0,:,0,2])


if __name__ == '__main__':
    main(parse_args())
