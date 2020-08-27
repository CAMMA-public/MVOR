# coding: utf-8
'''
File: eval_pck_2dpose.py
Project: CAMMA-MVOR
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import json
import numpy as np
import argparse
import sys

""" compute the pck evaluation metric between the ground truth camma json and detections json """

def coco_to_camma_kps(coco_kps):
    """
    convert coco keypoints(17) to camma-keypoints(10)
    :param coco_kps: 17 keypoints of coco
    :return: camma_keypoints
    """
    num_keypoints = 10
    camma_kps = np.empty((num_keypoints, 2))
    camma_kps[:] = np.nan

    nose = coco_kps[0, :]
    leye = coco_kps[1, :]
    reye = coco_kps[2, :]
    lear = coco_kps[3, :]
    rear = coco_kps[4, :]

    if np.all(np.logical_not(np.isnan(leye))) and np.all(np.logical_not(np.isnan(reye))):
        camma_kps[0, :] = ((leye + reye) / 2).reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(lear))) and np.all(np.logical_not(np.isnan(rear))):
        camma_kps[0, :] = ((lear + rear) / 2).reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(nose))):
        camma_kps[0, :] = nose.reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(reye))):
        camma_kps[0, :] = reye.reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(leye))):
        camma_kps[0, :] = leye.reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(rear))):
        camma_kps[0, :] = rear.reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(lear))):
        camma_kps[0, :] = lear.reshape(1, 2)

    lshoulder = coco_kps[5, :]
    rshoulder = coco_kps[6, :]
    lhip = coco_kps[11, :]
    rhip = coco_kps[12, :]
    lelbow = coco_kps[7, :]
    relbow = coco_kps[8, :]
    lwrist = coco_kps[9, :]
    rwrist = coco_kps[10, :]

    if np.all(np.logical_not(np.isnan(lshoulder))) and np.all(np.logical_not(np.isnan(rshoulder))):
        camma_kps[1, :] = ((lshoulder + rshoulder) / 2).reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(rshoulder))):
        camma_kps[1, :] = rshoulder.reshape(1, 2)
    elif np.all(np.logical_not(np.isnan(lshoulder))):
        camma_kps[1, :] = lshoulder.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(lshoulder))):
        camma_kps[2, :] = lshoulder.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(rshoulder))):
        camma_kps[3, :] = rshoulder.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(lhip))):
        camma_kps[4, :] = lhip.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(rhip))):
        camma_kps[5, :] = rhip.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(lelbow))):
        camma_kps[6, :] = lelbow.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(relbow))):
        camma_kps[7, :] = relbow.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(lwrist))):
        camma_kps[8, :] = lwrist.reshape(1, 2)

    if np.all(np.logical_not(np.isnan(rwrist))):
        camma_kps[9, :] = rwrist.reshape(1, 2)

    return camma_kps


def bestoverlapJoints(detJoints, gtbox, overlap):
    """
    compute the ovelrapping between the ground truth and detections
    :param detJoints:
    :param gtbox:
    :param overlap:
    :return: the detection with overlapping more than 0.3 and maximum score
    """
    box = np.array([])
    if detJoints.shape == 0:
        return box

    x1 = gtbox[0]
    y1 = gtbox[1]
    x2 = gtbox[2]
    y2 = gtbox[3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    bx = detJoints[:, 0::2]
    by = detJoints[:, 1::2]
    bx = bx[:, :-1]
    by = by[:, :-1]

    bx1 = np.nanmin(bx, 1)
    bx2 = np.nanmax(bx, 1)
    by1 = np.nanmin(by, 1)
    by2 = np.nanmax(by, 1)

    xx1 = np.maximum(x1, bx1)
    yy1 = np.maximum(y1, by1)
    xx2 = np.minimum(x2, bx2)
    yy2 = np.minimum(y2, by2)

    w = xx2 - xx1 + 1

    w[w < 0] = 0
    h = yy2 - yy1 + 1
    h[h < 0] = 0

    inter = w * h
    o = inter / area

    I = np.where(o > overlap)
    if not I[0].shape[0] == 0:
        scores1 = detJoints[:, -1]
        scores = scores1[I]
        maxscore = np.max(scores)
        index = np.where(np.abs(scores1 - maxscore) < 0.000001)
        index = index[0][0]
        box = detJoints[index, :]
    return box


def eval_pck(ca, gt, thresh=0.2, onlyVisible=False):
    """
    calculate pck results
    :param ca: detections list
    :param gt: ground truth list
    :param thresh
    :param onlyVisible: not used
    :return: pck array
    """
    assert len(ca.keys()) == len(gt.keys())

    # compute the scale of the ground truth results
    for key in gt:
        scale = np.max(np.nanmax(gt[key]['point'], 0) - np.nanmin(gt[key]['point'], 0) + 1)
        gt[key]['scale'] = scale

    tp = {}
    for n in gt:
        if not len(ca[n]['point'].shape) == 0:
            dist = np.sqrt(np.sum(np.square(ca[n]['point'] - gt[n]['point']), 1))
            dist[np.isnan(dist)] = np.finfo('float').max
            tp[n] = np.double(dist <= (thresh * gt[n]['scale']))
        else:
            tp[n] = np.zeros((gt[n]['point'].shape[0]))
    pck = np.zeros((gt[1]['point'].shape[0]))
    count = 0
    for p in tp:
        pck += tp[p]
        count += 1
    pck = pck / count

    return pck


def annolist2array(dt_annolist, gt_annolist):
    """
    convert the annolist to array
    :param inp_annolist:
    :return: the list
    """
    num_keypoints = 10
    num_keypoints_dt = 17
    i = 1
    list_gt = {}
    list_dt = {}

    # start iterating over the detections
    for imgidx, ann_dt in enumerate(dt_annolist):
        i_start = i
        num_gt_persons = len(gt_annolist[imgidx]['annorect']);
        for annidx in range(num_gt_persons):
            point = np.empty((num_keypoints, 2))
            point[:] = np.nan
            for p_idx in range(len(gt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'])):
                x = gt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['x'][0]
                y = gt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['y'][0]
                kp_id = gt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['id'][0]
                point[kp_id, 0] = x
                point[kp_id, 1] = y

            list_gt[i] = {}
            list_gt[i]['point'] = point
            i = i + 1

        num_dt_persons = len(dt_annolist[imgidx]['annorect'])
        detJoints = np.empty((num_dt_persons, 2 * num_keypoints_dt + 2))
        detJoints[:] = np.nan

        for annidx in range(num_dt_persons):
            score = dt_annolist[imgidx]['annorect'][annidx]['score'][0]
            x = np.empty((1, num_keypoints_dt))
            x[:] = np.nan
            y = np.empty((1, num_keypoints_dt))
            y[:] = np.nan
            for p_idx in range(len(dt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'])):
                x_p = dt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['x'][0]
                y_p = dt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['y'][0]
                kp_id = dt_annolist[imgidx]['annorect'][annidx]['annopoints'][0]['point'][p_idx]['id'][0]
                x[0, kp_id] = x_p
                y[0, kp_id] = y_p

            detJoints[annidx, 0::2][:-1] = x
            detJoints[annidx, 1::2][:-1] = y
            detJoints[annidx, -1] = score

        # print('gt size', len(list_gt.keys()))
        for annidx in range(num_gt_persons):
            x = list_gt[i_start]['point'][:, 0]
            y = list_gt[i_start]['point'][:, 1]
            x1 = x[np.logical_not(np.isnan(x))]
            y1 = y[np.logical_not(np.isnan(y))]

            if x1.shape and y1.shape:
                gtbox = [min(x1), min(y1), max(x1), max(y1)]
                box = bestoverlapJoints(detJoints, gtbox, 0.3)
                list_dt[i_start] = {}
                if not box.shape[0] == 0:
                    dt_point = np.zeros((num_keypoints_dt, 2))
                    # print('box[0, 0::2][:-1]', box[0::2][:-1].shape)
                    dt_point[:, 0] = box[0::2][:-1]
                    dt_point[:, 1] = box[1::2][:-1]
                    camma_dt_point = coco_to_camma_kps(dt_point)
                    list_dt[i_start]['point'] = camma_dt_point
                else:
                    list_dt[i_start]['point'] = np.empty([])
                i_start = i_start + 1

    return (list_dt, list_gt)


def get_all_anno(img_id, json_dict):
    anno = []
    for ann in json_dict['annotations']:
        if ann["image_id"] == img_id:
            anno.append(ann)
    return anno


def generate_gt_dict(camma_multiRGBD):
    annolist = []  ##one item per image

    for img in camma_multiRGBD['images']:
        annots = []  # all annotations of the given img
        original_annots = get_all_anno(img['id'], camma_multiRGBD)
        ann_idx = 0

        for original_ann in original_annots:
            point = []
            if original_ann['only_bbox'] == 0:
                x = original_ann["keypoints"][0::3]
                y = original_ann["keypoints"][1::3]
                for i in range(0, 10):
                    if not (x[i] == 0 and y[i] == 0):
                        kp_dict = {"x": [x[i]], "y": [y[i]], "id": [i]}
                        point.append(kp_dict)
                if np.count_nonzero(x) == 0 and np.count_nonzero(y) == 0:
                    continue
                annopoints = [{"point": point}]
                new_annot = {"annopoints": annopoints}
                annots.append(new_annot)
                ann_idx += 1
        img_dict = {"image": [{"name": img['id']}], "annorect": annots}
        annolist.append(img_dict)

    final_dict = {"annolist": annolist}
    return final_dict


def generate_new_annot(ann, kps):
    x = kps[0::3]
    y = kps[1::3]
    point = []
    for i in range(0, 10):
        if not (x[i] == 0 and y[i] == 0):
            kp_dict = {"x": [x[i]], "y": [y[i]], "id": [i]}
            point.append(kp_dict)
    annopoints = [{"point": point}]
    new_annot = {"score": ann['score'], "annopoints": annopoints}
    return new_annot


def get_image_id(image_file, gts_json):
    for image in gts_json['images']:
        if image['file_name'] == image_file:
            return image['id']


def generate_dt_dict(dts, gts):
    ##Input : dts and gts in coco format
    annolist = []

    for img in gts["images"]:
        img_dict = {"image": [{"name": img["id"]}], "annorect": []}
        annolist.append(img_dict)

    for dt in dts:
        img_id = dt["image_id"]
        x = dt["coco_keypoints"][0::3]
        y = dt["coco_keypoints"][1::3]
        score = dt["score"]

        for img_dict in annolist:
            if img_dict["image"][0]["name"] == img_id:
                point = []
                for i in range(0, 17):
                    if not (x[i] == 0 and y[i] == 0):
                        kp_dict = {"x": [x[i]], "y": [y[i]], "id": [i]}
                        point.append(kp_dict)
                annopoints = [{"point": point}]
                new_annotation = {"score": [score], "annopoints": annopoints}
                img_dict["annorect"].append(new_annotation)

    final_dict = {"annolist": annolist}
    return final_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the multiview OR CAMMA RGBD dataset')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='path to ground truth json file'
    )
    parser.add_argument(
        '--dt',
        type=str,
        default='',
        help='path to detection json file'
    )
    parser.add_argument(
        '--thresh',
        type=float,
        default=0.2,
        help='threshold value (default 0.2)'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    gt = args.gt
    dt = args.dt
    thresh = args.thresh

    with open(gt) as f:
        gt_json_coco = json.load(f)
    with open(dt) as f:
        dt_json_coco = json.load(f)

    gt_json = generate_gt_dict(gt_json_coco)
    gt_annolist = gt_json['annolist']

    dt_json = generate_dt_dict(dt_json_coco, gt_json_coco)
    dt_annolist = dt_json['annolist']

    list_dt, list_gt = annolist2array(dt_annolist, gt_annolist)
    #print('list_dt', len(list_dt))
    #print('list_gt', len(list_gt))

    pck = eval_pck(list_dt, list_gt, thresh)
    # compute the average
    pck = (pck + pck[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8]]) / 2.0
    pck = pck[[0, 2, 4, 6, 8]]

    mean_pck = np.mean(pck)

    pck = pck*100
    mean_pck = round(mean_pck*100, 1)
    np.set_printoptions(precision=1)
    print('  PCK Results ')
    print('[Head Shou  Hip Elbo Wris] [ mean--pck ]')
    print(pck, '[', mean_pck, ']')
    print('------------------------------------------------------------------')


if __name__ == '__main__':
    main(parse_args())
