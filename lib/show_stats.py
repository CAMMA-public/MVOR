# coding: utf-8
'''
File: show_stats.py
Project: CAMMA-MVOR
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import argparse
import json
import numpy as np
import sys


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_from_ann(gts_json, list_ann_id):
    number_persons = 0
    number_kps = np.zeros((10), dtype=float)
    if list_ann_id:
        for id in list_ann_id:
            for ann in gts_json['annotations']:
                if ann['id'] == id:
                    number_persons += 1
                    for i in range(10):
                        if ann['keypoints'][i * 3 + 2] > 1:
                            number_kps[i] += 1
    return number_persons, number_kps


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the multiview OR CAMMA RGBD dataset')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='path to ground truth json file containing 2D and 3D keypoints'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    if args.gt:
        with open(args.gt) as f:
            camma_multiRGBD = json.load(f)
    else:
        print('input json not given, exiting .')
        sys.exit(1)

    anno_2d = [p for p in camma_multiRGBD['annotations'] if not bool(p['only_bbox'])]
    sum_clini = sum(1 for p in camma_multiRGBD['annotations'] if p['person_role'] == 'clinician')
    sum_patient = sum(1 for p in camma_multiRGBD['annotations'] if p['person_role'] == 'patient')
    sum_face_vis = sum(1 for p in camma_multiRGBD['annotations'] if p['face_blurred'] == 1)
    print('----------------------------------------------------------------------------------------------------------')
    print('Number of multi-view frames = {}'.format(len(camma_multiRGBD['multiview_images'])))
    print('Number of person bounding boxes = {}'.format(len(camma_multiRGBD['annotations'])))
    print('Number of clinicians = {}'.format(sum_clini))
    print('Number of patients = {}'.format(sum_patient))
    print('Number of faces visible = {}'.format(sum_face_vis))
    print('Number of 2D keypoint annotations = {}'.format(len(anno_2d)))
    print('Number of 3D keypoint annotations = {}'.format(len(camma_multiRGBD['annotations3D'])))

    no_body_parts_3views = np.zeros((12), dtype=float)
    person_lengths = 0
    persons_visiblity_result = []
    for images in camma_multiRGBD['multiview_images']:
        (imgid0, imgid1, imgid2) = (images['images'][0]['id'], images['images'][1]['id'], images['images'][2]['id'])

        id_3d = str(imgid0) + '_' + str(imgid1) + '_' + str(imgid2)
        pers = [ann3d['person_id'] for ann3d in camma_multiRGBD['annotations3D'] if id_3d == ann3d['image_ids']]
        person_lengths += len(pers)
        annotations2d0 = [ann for ann in camma_multiRGBD['annotations']
                          if imgid0 == ann['image_id']]
        annotations2d1 = [ann for ann in camma_multiRGBD['annotations']
                          if imgid1 == ann['image_id']]
        annotations2d2 = [ann for ann in camma_multiRGBD['annotations']
                          if imgid2 == ann['image_id']]
        search_annotation = [annotations2d0, annotations2d1, annotations2d2]

        for index_person_id in pers:
            # index_person_id = ann['person_id']
            compute_stat_person = []
            # pq += 1
            for search_ann in search_annotation[0]:
                if search_ann['person_id'] == index_person_id:
                    compute_stat_person.append(search_ann)

            for search_ann1 in search_annotation[1]:
                if search_ann1['person_id'] == index_person_id:
                    compute_stat_person.append(search_ann1)

            for search_ann2 in search_annotation[2]:
                if search_ann2['person_id'] == index_person_id:
                    compute_stat_person.append(search_ann2)

            if len(compute_stat_person) == 0:
                print('no views still 3d', index_person_id, imgid0, imgid1, imgid2)
                continue

            ids_f = [p['id'] for p in compute_stat_person]
            _, num_kps = compute_from_ann(camma_multiRGBD, ids_f)
            persons_visiblity_result.append((len(compute_stat_person), num_kps, id_3d))

    nbp_view3, nbp_view2, nbp_view1 = np.zeros((10), dtype=float), np.zeros((10), dtype=float), np.zeros((10),
                                                                                                         dtype=float)
    np3, np2, np1 = 0, 0, 0
    for person_vis in persons_visiblity_result:
        if person_vis[0] == 3:
            np3 += 1
            nbp_view3 += person_vis[1]

        elif person_vis[0] == 2:
            np2 += 1
            nbp_view2 += person_vis[1]

        elif person_vis[0] == 1:
            np1 += 1
            nbp_view1 += person_vis[1]

    v3 = (nbp_view3 / 3.0).astype(int)
    v2 = (nbp_view2 / 2.0).astype(int)
    v1 = (nbp_view1 / 1.0).astype(int)
    print('person[', 'Head', 'Neck', 'Shoulder-L', 'Shoulder-R', 'Hip-L',
          'Hip-R', 'Elbow-L', 'Elbow-R', 'Wrist-L', 'Wrist-R]')
    print(np3, '  [', v3[0], ' ', v3[1], '  ', v3[2], '     ', v3[3], '     ', v3[4], '  ',
          v3[5], '', v3[6], '    ', v3[7], '    ', v3[8], '  ', v3[9],
          ' ]=> No of person visible in all 3 views with each body part visibility')
    print(np2, '  [', v2[0], ' ', v2[1], '  ', v2[2], '     ', v2[3], '     ', v2[4], '  ',
          v2[5], '', v2[6], '    ', v2[7], '    ', v2[8], '  ', v2[9],
          ' ]=> No of person visible 2 views with each body part visibility')
    print(np1, '  [', v1[0], ' ', v1[1], '  ', v1[2], '     ', v1[3], '     ', v1[4], '  ',
          v1[5], '', v1[6], '     ', v1[7], '     ', v1[8], '   ', v1[9],
          '  ]=> No of person visible in 1 view with each body part visibility')
    print('----------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main(parse_args())
