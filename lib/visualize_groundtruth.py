# coding: utf-8
'''
File: visualize_groundtruth.py
Project: CAMMA-MVOR
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import argparse
import json
import cv2
import os
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from IPython.display import HTML, display

plt.ion()

color_brightness = 150
cc = {}
cc['r'] = (0, 0, color_brightness)
cc['g'] = (0, color_brightness, 0)
cc['b'] = (color_brightness, 0, 0)
cc['c'] = (color_brightness, color_brightness, 0)
cc['m'] = (color_brightness, 0, color_brightness)
cc['y'] = (0, color_brightness, color_brightness)
cc['w'] = (color_brightness, color_brightness, color_brightness)
cc['k'] = (0, 0, 0)
cc['t1'] = (205, 97, 85)
cc['t2'] = (33, 97, 140)
cc['t3'] = (23, 165, 137)
cc['t4'] = (125, 102, 8)
cc['t5'] = (230, 126, 34)
cc['t6'] = (211, 84, 0)
cc['t7'] = (52, 73, 94)
cc['t8'] = (102, 255, 153)
cc['t9'] = (51, 0, 204)
cc['t10'] = (255, 0, 204)

cir_radius = 3
max_persons = 100

""" predefined first 20 colors """
colors_arr = {}
colors_arr[0] = (0, 0, color_brightness)
colors_arr[1] = (0, color_brightness, 0)
colors_arr[2] = (color_brightness, 0, 0)
colors_arr[3] = (color_brightness, color_brightness, 0)
colors_arr[4] = (color_brightness, 0, color_brightness)
colors_arr[5] = (0, color_brightness, color_brightness)
colors_arr[6] = (color_brightness, color_brightness / 2, 255 - color_brightness / 2)
colors_arr[7] = (color_brightness / 2, 255 - color_brightness / 2, color_brightness)
colors_arr[8] = (color_brightness, color_brightness / 2, 255 - color_brightness / 2)
colors_arr[9] = (color_brightness / 2, 255 - color_brightness / 2, color_brightness / 2)
colors_arr[10] = (0, 0, color_brightness / 2)
colors_arr[11] = (0, color_brightness / 2, 0)
colors_arr[12] = (color_brightness / 2, 0, 0)
colors_arr[13] = (color_brightness / 2, color_brightness / 2, 0)
colors_arr[14] = (color_brightness / 2, 0, color_brightness / 2)
colors_arr[15] = (0, color_brightness / 2, color_brightness / 2)
colors_arr[16] = (color_brightness / 2, color_brightness / 4, 255 - color_brightness / 4)
colors_arr[17] = (color_brightness / 4, 255 - color_brightness / 4, color_brightness / 2)
colors_arr[18] = (color_brightness / 2, color_brightness / 4, 255 - color_brightness / 4)
colors_arr[19] = (color_brightness / 4, 255 - color_brightness / 4, color_brightness / 4)
camma_colors = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
camma_colors_float = [(0.7, 0.7, 0.0, 0.5), (0.0, 0.7, 0.0, 0.5), (0.7, 0.0, 0.7, 0.5),
                      (0.0, 0.7, 0.0, 0.5), (0.7, 0.0, 0.7, 0.5), (0.0, 0.7, 0.0, 0.5),
                      (0.7, 0.0, 0.7, 0.5), (0.0, 0.7, 0.0, 0.5), (0.7, 0.0, 0.7, 0.5),
                      (0.0, 0.7, 0.0, 0.5)]
camma_pairs = [[1, 2], [2, 4], [4, 8], [4, 6], [8, 10], [2, 3],
               [3, 5], [3, 7], [7, 9], [5, 6]]
camma_colors_skeleton = ['y', 'g', 'g', 'g', 'g', 'm', 'm', 'm', 'm', 'm']
camma_part_names = ['head', 'neck', 'left_shoulder', 'right_shoulder', 'left_hip',
                    'right_hip', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']


def str2bool(v):
    """
    convert string to boolean (string passed through command line arguments)
    :param v: string argument ('yes', 'true', 't', 'y', '1' => for logical true;
                                'no', 'false', 'f', 'n', '0' => for logical false)
    :return: boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """
    parse the command line arguments run python visualize_groundtruth.py -h to see command line options
    :return:
    """
    parser = argparse.ArgumentParser(description='Visualize the multiview OR CAMMA RGBD dataset')
    parser.add_argument(
        '--inp_json',
        type=str,
        default='',
        help='path to input json file containing 2D and 3D keypoints'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default='',
        help='path to the image directory'
    )
    parser.add_argument(
        '--viz_attr',
        type=str,
        default="false",
        help='visualize the annotation attributes such as person role, joint names etc'
    )
    parser.add_argument(
        '--viz_3D',
        type=str,
        default="false",
        help='visualize the 3D annotations'
    )
    parser.add_argument(
        '--viz_2D',
        type=str,
        default="false",
        help='visualize the 2D annotations'
    )
    parser.add_argument(
        '--show_ann',
        type=str,
        default="true",
        help='Show the annotations'
    )
    parser.add_argument(
        '--show_3dto2dproj',
        type=str,
        default="true",
        help='Show the 3D to 2D projections on the 2D images'
    )
    parser.add_argument(
        '--show_pose_variability',
        type=str,
        default="false",
        help='Show the pose variability of the 2D annotations'
    )
    parser.add_argument(
        '--save_gt',
        type=str,
        default='',
        help='it will not render anything on the screen but write the visualizations on the given path'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def coco_to_camma_kps(coco_kps):
    """
    convert coco keypoints(17) to camma-keypoints(10)
    :param coco_kps: 17 keypoints of coco
    :return: camma_keypoints
    """
    num_keypoints = 10
    camma_kps = np.zeros((num_keypoints, 3))

    nose = coco_kps[0, :]
    leye = coco_kps[1, :]
    reye = coco_kps[2, :]
    lear = coco_kps[3, :]
    rear = coco_kps[4, :]

    if leye[-1] > 0 and reye[-1] > 0:
        camma_kps[0, :] = ((leye + reye) / 2).reshape(1, 3)
    elif lear[-1] > 0 and rear[-1] > 0:
        camma_kps[0, :] = ((lear + rear) / 2).reshape(1, 3)
    elif nose[-1] > 0:
        camma_kps[0, :] = nose.reshape(1, 3)
    elif reye[-1] > 0:
        camma_kps[0, :] = reye.reshape(1, 3)
    elif leye[-1] > 0:
        camma_kps[0, :] = leye.reshape(1, 3)
    elif rear[-1] > 0:
        camma_kps[0, :] = rear.reshape(1, 3)
    elif lear[-1] > 0:
        camma_kps[0, :] = lear.reshape(1, 3)
    else:
        camma_kps[0, :] = np.array([0, 0, 0]).reshape(1, 3)

    lshoulder = coco_kps[5, :]
    rshoulder = coco_kps[6, :]
    lhip = coco_kps[11, :]
    rhip = coco_kps[12, :]
    lelbow = coco_kps[7, :]
    relbow = coco_kps[8, :]
    lwrist = coco_kps[9, :]
    rwrist = coco_kps[10, :]

    if lshoulder[-1] > 0 and rshoulder[-1] > 0:
        camma_kps[1, :] = ((lshoulder + rshoulder) / 2).reshape(1, 3)
    elif rshoulder[-1] > 0:
        camma_kps[1, :] = rshoulder.reshape(1, 3)
    elif lshoulder[-1] > 0:
        camma_kps[1, :] = lshoulder.reshape(1, 3)
    else:
        camma_kps[1, :] = np.array([0, 0, 0]).reshape(1, 3)

    if lshoulder[-1] > 0:
        camma_kps[2, :] = lshoulder.reshape(1, 3)

    if rshoulder[-1] > 0:
        camma_kps[3, :] = rshoulder.reshape(1, 3)

    if lhip[-1] > 0:
        camma_kps[4, :] = lhip.reshape(1, 3)

    if rhip[-1] > 0:
        camma_kps[5, :] = rhip.reshape(1, 3)

    if lelbow[-1] > 0:
        camma_kps[6, :] = lelbow.reshape(1, 3)

    if relbow[-1] > 0:
        camma_kps[7, :] = relbow.reshape(1, 3)

    if lwrist[-1] > 0:
        camma_kps[8, :] = lwrist.reshape(1, 3)

    if rwrist[-1] > 0:
        camma_kps[9, :] = rwrist.reshape(1, 3)

    return camma_kps



def viz2d(inp, annotations, viz_attributes=False, viz_projection=False):
    """
    2D annotations visualization
    :param im: input image
    :param annotations: ground truth annotations
    :param viz_attributes: whether to show the attributes
    :param viz_projection: whether to show the 3D to 2D projections
    :return: image with rendered anootations
    """
    im = inp.copy()
    countPerson = 0
    height, width, layers = im.shape
    if not viz_projection:
        for person in annotations:
            rect = person['bbox']
            cv2.rectangle(im, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                          colors_arr[countPerson], thickness=1, lineType=cv2.LINE_AA)
            if not bool(person['only_bbox']):
                cv2.rectangle(im, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                              colors_arr[countPerson], thickness=1, lineType=cv2.LINE_AA)
                pose = np.array(person['keypoints']).reshape(-1, 3)[:, :3]
                cv2.rectangle(im, (int(rect[0]), int(rect[1] + 13)),
                              (int(rect[0] + rect[2]), int(rect[1])),
                              (0, 0, 0), thickness=-1)
                if viz_attributes:
                    person_role = person['person_role']
                    str_disp = 'person(' + str(person['person_id']) + ')(' + person_role + ')'
                else:
                    # str_disp = 'person(' + str(person['person_id']) + ')' + '(' + person['id'] + ')'
                    str_disp = 'person(' + str(person['person_id']) + ')'

                cv2.putText(im, str_disp, (int(rect[0]), int(rect[1] + 10)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if pose.shape[0] == 10:
                    for idx in range(len(camma_colors_skeleton)):
                        pt1 = (int(np.clip(pose[camma_pairs[idx][0] - 1, 0], 0, width)),
                               int(np.clip(pose[camma_pairs[idx][0] - 1, 1], 0, height)))
                        pt2 = (int(np.clip(pose[camma_pairs[idx][1] - 1, 0], 0, width)),
                               int(np.clip(pose[camma_pairs[idx][1] - 1, 1], 0, height)))
                        if 0 not in pt1 + pt2:
                            cv2.line(im, pt1, pt2, cc[camma_colors_skeleton[idx]], 3, cv2.LINE_AA)
                    """ draw the skelton points """
                    for idx_c, color in enumerate(camma_colors):
                        pt = (int(np.clip(pose[idx_c, 0], 0, width)),
                              int(np.clip(pose[idx_c, 1], 0, height)))
                        if 0 not in pt:
                            cv2.circle(im, pt, 3, cc[color], 2, cv2.LINE_AA)
                            if viz_attributes:
                                cv2.putText(im, camma_part_names[idx_c], pt, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255),
                                            2)
                countPerson += 1
    else:
        pose = annotations[:, 0:10]
        if pose.shape[1] == 10:
            for idx in range(len(camma_colors_skeleton)):
                pt1 = (int(np.clip(pose[0, camma_pairs[idx][0] - 1], 0, width)),
                       int(np.clip(pose[1, camma_pairs[idx][0] - 1], 0, height)))
                pt2 = (int(np.clip(pose[0, camma_pairs[idx][1] - 1], 0, width)),
                       int(np.clip(pose[1, camma_pairs[idx][1] - 1], 0, height)))
                if 0 not in pt1 + pt2:
                    cv2.line(im, pt1, pt2, (200, 200, 200), 1, cv2.LINE_AA)
            """ draw the skelton points """
            for idx_c, color in enumerate(camma_colors):
                pt = (int(np.clip(pose[0, idx_c], 0, width)),
                      int(np.clip(pose[1, idx_c], 0, height)))
                if 0 not in pt:
                    cv2.circle(im, pt, 2, cc[color], 2, cv2.LINE_AA)
    return im


def transformGivenTrfMatrix(ann3d, tr_mat):
    """
    Transform the 3D points from one coordiate system to another given 4x4 transformation matrix
    :param ann3d: input 3D points
    :param tr_mat: 4x4 transformation matrix
    :return: transformed 3D points
    """
    X = ann3d['keypoints3D'][0::4]
    Y = ann3d['keypoints3D'][1::4]
    Z = ann3d['keypoints3D'][2::4]
    pt3d = np.vstack((X, Y, Z, np.ones(len(X))))
    # transform points to room coordinate
    pt3d = np.dot(tr_mat, pt3d)
    pt3d = pt3d[0:3]
    return pt3d


def projectCam3DTo2D(pose3D, camparam):
    """
    Project 3D point in camera coordinates to image given intrinsic camera parameters (focal-length and principal-point)
    :param pose3D: 3D points in camera coordinates
    :param camparam: intrinsic camera parameters
    :return: 2D point on the images
    """
    focal = camparam['focallength']
    pp = camparam['principalpoint']
    pose3D[2][pose3D[2] == 0.0] = 1.0  # replace zero with 1 to avoid divide by zeros
    p1 = ((np.divide(pose3D[0], pose3D[2])) * focal[0]) + pp[0]
    p2 = ((np.divide(pose3D[1], pose3D[2])) * focal[1]) + pp[1]
    return np.vstack((p1, p2))


def createopencvwindows():
    """
    create opencv windows to display images
    :return:
    """
    cv2.namedWindow("cam-1")
    cv2.moveWindow("cam-1", 0, 0)

    cv2.namedWindow("cam-2")
    cv2.moveWindow("cam-2", 660, 0)

    cv2.namedWindow("cam-3")
    cv2.moveWindow("cam-3", 320, 540)


def create_index(camma_mvor_gt):
    """
    get the 2D and 3D annotations for each image from the coco style annotations
    :param camma_mvor_gt: ground truth dict
    :return: 2D and 3D annotations dictionary; key=image_id, value = 2D or 3D annotations
    """
    im_ids_2d = [p['id'] for p in camma_mvor_gt['images']]
    im_ids_3d = [p['id'] for p in camma_mvor_gt['multiview_images']]
    mv3d_paths = {p["id"]:p["images"] for p in camma_mvor_gt['multiview_images']}
    imid_to_path = {p["id"]:p["file_name"] for p in camma_mvor_gt['images']}

    anns_2d = {str(key): [] for key in im_ids_2d}
    anns_3d = {str(key): [] for key in im_ids_3d}
    print('creating index for 2D annotations')
    for ann in camma_mvor_gt['annotations']:
        anns_2d[str(ann['image_id'])].append({
            'keypoints': ann['keypoints'],
            'bbox': ann['bbox'],
            'person_id': ann['person_id'],
            'person_role': ann['person_role'],
            'only_bbox': ann['only_bbox'],
            'id': ann['id']
        })
    print('done')
    print('creating index for 3D annotations')
    for ann3d in camma_mvor_gt['annotations3D']:
        anns_3d[str(ann3d['image_ids'])].append({
            'id': ann3d['id'],
            'person_id': ann3d['person_id'],
            'ref_camera': ann3d['ref_camera'],
            'keypoints3D': ann3d['keypoints3D']
        })
    print('Index creation done')

    return anns_2d, anns_3d, mv3d_paths, imid_to_path

def progress_bar(value, max=100):
    """ A HTML helper function to display the progress bar
    Args:
        value ([int]): [current progress bar value]
        max (int, optional): [maximum value]. Defaults to 100.
    Returns:
        [str]: [HTML progress bar string]
    """
    return HTML(
        """
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(
            value=value, max=max
        )
    )

def bgr2rgb(im):
    """[convert opencv image in BGR format to RGB format]
    Args:
        im ([numpy.ndarray]): [input image in BGR format]
    Returns:
        [numpy.ndarray]: [output image in RGB format]
    """    
    b, g, r = cv2.split(im)
    return cv2.merge([r, g, b])


def plt_imshow(im, title=''):
    """
    show the 2D plot using matplotlib
    :param im: input image
    :param title: title of the plot
    :return:
    """
    im = bgr2rgb(im)
    if title:
        plt.title(title, fontsize=7)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplots_adjust(left=0.01, bottom=0.01, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
    plt.imshow(im)


def rect_prism(ax, x_range, y_range, z_range):
    """
    plot the 3d rectangle
    :param x_range:
    :param y_range:
    :param z_range:
    :return:
    """
    xx, yy = np.meshgrid(x_range, y_range)
    zz = z_range[0] * np.ones(xx.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)

    zz = z_range[1] * np.ones(xx.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)

    yy, zz = np.meshgrid(y_range, z_range)
    xx = x_range[0] * np.ones(yy.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)

    xx = x_range[1] * np.ones(yy.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)

    xx, zz = np.meshgrid(x_range, z_range)
    yy = y_range[0] * np.ones(zz.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)

    yy = y_range[1] * np.ones(zz.shape)
    ax.plot_wireframe(xx, yy, zz, color="g")
    ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)


def plt_3dplot(fig, anns3d, camma_mvor_gt):
    """
    plot the 3D annotations on the given fig
    :param fig: input figure
    :param anns3d: 3D annotations
    :param camma_mvor_gt: ground truth dictionary
    :return:
    """
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.tick_params()
    ax.tick_params(labelsize=3)
    plt.title('3D view', fontsize=7)
    for ann3d in anns3d:
        tr_mat = np.array(camma_mvor_gt['cameras_info']['camParams'][
                              'firstCamToRoomRef']).reshape((4, 4))
        pt3d = transformGivenTrfMatrix(ann3d, tr_mat)
        pers_id = ann3d['person_id']
        for idx in range(len(camma_colors_skeleton)):
            pt1 = pt3d[:, camma_pairs[idx][0] - 1][0:3]
            pt2 = pt3d[:, camma_pairs[idx][1] - 1][0:3]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], camma_colors_skeleton[idx], zs=[pt1[2], pt2[2]],
                    alpha=0.7, linewidth=0.7)

        # plot 3D circles
        for idx in range(len(camma_colors)):
            pt = pt3d[:, idx][0:3]
            ax.scatter(pt[0], pt[1], pt[2], s=10, c=camma_colors_float[idx], marker='o')
        # plot text
        pt = pt3d[:, 0][0:3]
        # str_per = 'person(' + str(pers_id) + ')'
        # ax.text(pt[0], pt[1], pt[2], str(pers_id) + '(' + ann3d['id'] + ')', 'y', fontsize=10)
        ax.text(pt[0], pt[1], pt[2], str(pers_id), 'y', fontsize=6)
    # rect_prism(ax, np.array([-1000, -500]), np.array([750, 1500]), np.array([-10, 10]))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-200, 2500)
    ax.set_zlim(-200, 600)
    # plt.subplots_adjust(left=0.01, bottom=0.01, right=1.0, top=1.0, wspace=0.01, hspace=0.01)


def visualize_pose_variability(gt):
    """
    :param gt: ground truth
    :return:
    ##  0     1      2        3       4     5     6     7     8     9
    ## Head, Neck, LShould, Rshould, LHip, Rhip, Lelb, Relb, LWri, Rwri
    """
    joints = [[0, 1], [2, 6], [6, 8], [3, 7], [7, 9], [1, 10]]
    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (255, 255, 51), (0, 0, 255), (0, 255, 255)]
    default_kps = np.array([250, 150, 1,
                            250, 250, 1,
                            270, 280, 1,
                            230, 280, 1,
                            350, 430, 1,
                            150, 430, 1,
                            350, 330, 1,
                            150, 330, 1,
                            370, 390, 1,
                            130, 390, 1])
    centered_kps = []
    for ann in gt['annotations']:
        if 'keypoints' in ann.keys():
            kps = ann['keypoints']
            if np.count_nonzero(kps) > 0:
                neck = np.array(kps[3:5])
                diff = np.array([250, 250]) - neck
                new_kps = np.zeros(10 * 3, dtype=int)
                for i in range(0, 10):
                    new_kps[i * 3] = kps[i * 3] + diff[0]
                    new_kps[i * 3 + 1] = kps[i * 3 + 1] + diff[1]
                    new_kps[i * 3 + 2] = kps[i * 3 + 2]
                kps = [int(kp) for kp in kps]
                centered_kps.append(new_kps)

    img = np.ones((500, 500, 3), np.uint8) * 255
    for kps in centered_kps:
        for j, jt in enumerate(joints):
            if jt[1] == 10:  ##joint is neck -> center hip
                center_hip = (kps[4 * 3:4 * 3 + 2] + kps[5 * 3:5 * 3 + 2]) / 2
                center_hip = [int(pt) for pt in center_hip]
                if kps[jt[0] * 3 + 2] > 0 and kps[4 * 3 + 2] > 0 and kps[5 * 3 + 2]:
                    cv2.line(img, (center_hip[0], center_hip[1]), (kps[3], kps[4]), colors[j], 1,
                             lineType=cv2.LINE_AA)
            else:
                if kps[jt[0] * 3 + 2] > 0 and kps[jt[1] * 3 + 2] > 0:
                    cv2.line(img, (kps[jt[0] * 3], kps[jt[0] * 3 + 1]), (kps[jt[1] * 3], kps[jt[1] * 3 + 1]), colors[j],
                             1, lineType=cv2.LINE_AA)

    img_bodyparts = np.ones((500, 500, 3), np.uint8) * 255
    for j, jt in enumerate(joints):
        if jt[1] == 10:
            cv2.line(img_bodyparts, (250, 280), (250, 400), colors[j], 10,
                     lineType=cv2.LINE_AA)

        else:
            cv2.line(img_bodyparts, (default_kps[jt[0] * 3], default_kps[jt[0] * 3 + 1]),
                     (default_kps[jt[1] * 3], default_kps[jt[1] * 3 + 1]),
                     colors[j], 10, lineType=cv2.LINE_AA)
    cv2.putText(img_bodyparts, 'Color coding of the body parts',
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 134, 0), 1)
    cv2.putText(img, 'Pose variablity of the 2D annotations', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23, 134, 0), 1)
    img = np.hstack((img_bodyparts, img))
    #cv2.imwrite('pose_variability.png', img)
    cv2.imshow("Visualization of upper body pose variability", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    # Get the command line arguments
    inp_json = args.inp_json
    img_dir = args.img_dir
    viz_attr = str2bool(args.viz_attr)
    viz_3d = str2bool(args.viz_3D)
    viz_2d = str2bool(args.viz_2D)
    show_ann = str2bool(args.show_ann)
    save_gt = args.save_gt
    show_3dto2dproj = str2bool(args.show_3dto2dproj)
    show_pos_var = str2bool(args.show_pose_variability)

    if not img_dir:
        print('image dir missing')
        sys.exit(1)

    # Get the ground truth annotations
    if inp_json:
        with open(inp_json) as f:
            camma_mvor_gt = json.load(f)
    else:
        print('input json not given. Exiting...')
        sys.exit(1)

    if show_pos_var:
        visualize_pose_variability(camma_mvor_gt)

    if show_ann:
        # iterate over images
        # create the index
        anno_2d, anno3d = create_index(camma_mvor_gt)
        if viz_3d:
            fig = plt.figure(figsize=plt.figaspect(0.75))
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 1280
            fig_size[1] = 1000
            plt.rcParams["figure.figsize"] = fig_size
        else:
            createopencvwindows()
        # count3d = 0
        for index, images in enumerate(camma_mvor_gt['multiview_images']):
            (img1, img2, img3) = (images['images'][0], images['images'][1], images['images'][2])
            im1 = cv2.imread(os.path.join(img_dir, img1['file_name']))
            im2 = cv2.imread(os.path.join(img_dir, img2['file_name']))
            im3 = cv2.imread(os.path.join(img_dir, img3['file_name']))
            imgid1, imgid2, imgid3 = images['images'][0]['id'], images['images'][1]['id'], images['images'][2]['id']
            imid3d = str(imgid1) + '_' + str(imgid2) + '_' + str(imgid3)

            anns2d1, anns2d2, anns2d3 = anno_2d[str(imgid1)], anno_2d[str(imgid2)], anno_2d[str(imgid3)]
            anns3d = anno3d[imid3d]
            if viz_2d:
                im1 = viz2d(im1, anns2d1, viz_attr)
                im2 = viz2d(im2, anns2d2, viz_attr)
                im3 = viz2d(im3, anns2d3, viz_attr)
            if show_3dto2dproj:
                for ann3d in anns3d:
                    tr_mat0 = np.array(camma_mvor_gt['cameras_info']['camParams'][
                                           'extrinsics'][0]).reshape((4, 4))
                    tr_mat1 = np.array(camma_mvor_gt['cameras_info']['camParams'][
                                           'extrinsics'][1]).reshape((4, 4))
                    tr_mat2 = np.array(camma_mvor_gt['cameras_info']['camParams'][
                                           'extrinsics'][2]).reshape((4, 4))
                    pt3d0 = transformGivenTrfMatrix(ann3d, np.linalg.inv(tr_mat0))
                    pt3d1 = transformGivenTrfMatrix(ann3d, np.linalg.inv(tr_mat1))
                    pt3d2 = transformGivenTrfMatrix(ann3d, np.linalg.inv(tr_mat2))
                    pt2d0 = projectCam3DTo2D(pt3d0, camma_mvor_gt['cameras_info']['camParams']['intrinsics'][0])
                    pt2d1 = projectCam3DTo2D(pt3d1, camma_mvor_gt['cameras_info']['camParams']['intrinsics'][1])
                    pt2d2 = projectCam3DTo2D(pt3d2, camma_mvor_gt['cameras_info']['camParams']['intrinsics'][2])
                    im1 = viz2d(im1, pt2d0, viz_projection=True)
                    im2 = viz2d(im2, pt2d1, viz_projection=True)
                    im3 = viz2d(im3, pt2d2, viz_projection=True)
            if not viz_3d:
                # just show 2D annotations
                if save_gt:
                    if not os.path.isdir(os.path.join(save_gt, 'render')):
                        os.makedirs(os.path.join(save_gt, 'render'))
                    h1, w1 = im1.shape[:2]
                    h2, w2 = im2.shape[:2]
                    h3, w3 = im3.shape[:2]
                    vis = np.zeros((max(h1, h2), w1 + w2 + w3, 3), np.uint8)
                    vis[:h1, :w1, :3] = im1
                    vis[:h2, w1:w1 + w2, :3] = im2
                    vis[:h2, w1 + w2:w1 + w2 + w3, :3] = im3
                    cv2.imwrite(os.path.join(save_gt, 'render/' + imid3d + '.png'), vis)
                    print('saving image =>', os.path.join(save_gt, 'render/' + imid3d + '.png'))
                else:
                    cv2.imshow("cam-1", im1)
                    cv2.imshow("cam-2", im2)
                    cv2.imshow("cam-3", im3)
                    k = cv2.waitKey(0)
                    if k == 27 or k == ord('q'):  # wait for ESC or q key to exit
                        cv2.destroyAllWindows()
                        break
            else:
                # show 2D as well as 3D annotations
                fig.add_subplot(2, 2, 1)
                plt_imshow(im1, 'cam-1')

                fig.add_subplot(2, 2, 2)
                plt_imshow(im2, 'cam-2')

                fig.add_subplot(2, 2, 3)
                plt_imshow(im3, 'cam-3')

                # 3D plot
                plt_3dplot(fig, anns3d, camma_mvor_gt)
                if save_gt:
                    if not os.path.isdir(os.path.join(save_gt, 'render')):
                        os.makedirs(os.path.join(save_gt, 'render'))
                    extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(os.path.join(save_gt, 'render/' + imid3d + '.png'), pad_inches=0.0,
                                bbox_inches=extent, dpi=200)
                    print('saving image =>', os.path.join(save_gt, 'render/' + imid3d + '.png'))
                else:
                    plt.show()
                    break
                    plt.waitforbuttonpress(-1)
    else:
        # just show the images without any annotations
        createopencvwindows()
        for images in camma_mvor_gt['multiview_images']:
            (img1, img2, img3) = (images['images'][0], images['images'][1], images['images'][2])
            cv2.imshow("cam-1", cv2.imread(os.path.join(img_dir, img1['file_name'])))
            cv2.imshow("cam-2", cv2.imread(os.path.join(img_dir, img2['file_name'])))
            cv2.imshow("cam-3", cv2.imread(os.path.join(img_dir, img3['file_name'])))
            k = cv2.waitKey(0)
            if k == 27 or k == ord('q'):  # wait for ESC or q key to exit
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    main(parse_args())
