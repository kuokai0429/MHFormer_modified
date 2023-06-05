# 2023.0605 @Brian
# Visualization for Benchmarking.

import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import shutil
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed
from scipy.spatial.transform import Rotation as R

sys.path.append(os.getcwd())
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(keypoints_3d_gt, keypoints_3d_mhformer, keypoints_3d_poseformer, output_dir):

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    print('\nGenerating Ground Truth 3D pose...')

    for i in tqdm(range(len(keypoints_3d_gt[:]))):

        # Rotate vector(s) v about the rotation described by quaternion(s) q (Quaternion-derived rotation matrix): https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        rot = R.from_quat([0, 0, 0, 1]).as_quat()
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(keypoints_3d_gt[i], R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose( post_out, ax)

        output_dir_3D = output_dir +'GroundTruth_pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close(fig)

    print('\nGenerating MHFormer Predicted 3D pose...')

    for i in tqdm(range(len(keypoints_3d_mhformer[:]))):
        
        # Rotate vector(s) v about the rotation described by quaternion(s) q (Quaternion-derived rotation matrix): https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        rot = [0.0, 0.0, 0.0, 0.0]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(keypoints_3d_mhformer[i], R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose( post_out, ax)

        output_dir_3D = output_dir +'Predicted_pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close(fig)

    print('\nGenerating PoseFormer Predicted 3D pose...')

    # for i in tqdm(range(len(keypoints_3d_pred[:]))):
        
    #     # Rotate vector(s) v about the rotation described by quaternion(s) q (Quaternion-derived rotation matrix): https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    #     rot = [0.0, 0.0, 0.0, 0.0]
    #     rot = np.array(rot, dtype='float32')
    #     post_out = camera_to_world(keypoints_3d_pred[i], R=rot, t=0)
    #     post_out[:, 2] -= np.min(post_out[:, 2])

    #     fig = plt.figure( figsize=(9.6, 5.4))
    #     gs = gridspec.GridSpec(1, 1)
    #     gs.update(wspace=-0.00, hspace=0.05) 
    #     ax = plt.subplot(gs[0], projection='3d')
    #     show3Dpose( post_out, ax)

    #     output_dir_3D = output_dir +'Predicted_pose3D/'
    #     os.makedirs(output_dir_3D, exist_ok=True)
    #     plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
    #     plt.close(fig)
        
    print('Generating 3D pose successfully!')

    print('\nGenerating Benchmark visualizations...')

    # image_dir = 'results/' 
    # image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    # image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    # for i in tqdm(range(len(image_2d_dir))):
    #     image_2d = plt.imread(image_2d_dir[i])
    #     image_3d = plt.imread(image_3d_dir[i])

    #     ## crop
    #     edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
    #     image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

    #     edge = 130
    #     image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

    #     ## show
    #     font_size = 12
    #     fig = plt.figure(figsize=(9.6, 5.4))
    #     ax = plt.subplot(121)
    #     showimage(ax, image_2d)
    #     ax.set_title("Input", fontsize = font_size)

    #     ax = plt.subplot(122)
    #     showimage(ax, image_3d)
    #     ax.set_title("Ground Truth", fontsize = font_size)

    #     ## save
    #     output_dir_pose = output_dir +'pose/'
    #     os.makedirs(output_dir_pose, exist_ok=True)
    #     plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
    #     plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='S11', help='Human3.6M Subject.')
    parser.add_argument('--action', type=str, default='Walking', help='Human3.6M Action.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    video_name = f"{args.subject}_{args.action}"
    output_dir = f'./demo/output_benchmark/{args.subject}_{args.action}/'
    print(output_dir)

    # Ground Truth
    data = np.load('dataset/data_3d_h36m.npz', allow_pickle=True)
    # print(data.files, dict(enumerate(data['positions_3d'].flatten()))[0].keys(), dict(enumerate(data['positions_3d'].flatten()))[0]['S11'].keys())
    # print("Walking", len(dict(enumerate(data['positions_3d'].flatten()))[0]['S11']['Walking']))
    # print("Sitting 1", len(dict(enumerate(data['positions_3d'].flatten()))[0]['S11']['Sitting 1']))
    # print("Phoning 2", len(dict(enumerate(data['positions_3d'].flatten()))[0]['S11']['Phoning 2']))
    # print("Greeting 2", len(dict(enumerate(data['positions_3d'].flatten()))[0]['S11']['Greeting 2']))
    target = dict(enumerate(data['positions_3d'].flatten()))[0][args.subject][args.action]
    temp = []
    for j in range(0, len(target), 5):
        temp.append([target[j][i] for i in range(32) if i not in [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]])
    target = torch.Tensor(np.asarray(temp))
    # print(len(target))

    # Predicted (MHFormer) 
    data = np.load(f'demo/output/{args.subject}_{args.action}/keypoints_3d.npz', allow_pickle=True)
    predicted_mhf = torch.Tensor(data['reconstruction'])
    # print(len(predicted_mhf))

    # Predicted (PoseFormer) 
    data = np.load(f'demo/output/{args.subject}_{args.action}/keypoints_3d.npz', allow_pickle=True)
    predicted_pf = torch.Tensor(data['reconstruction'])
    # print(len(predicted_pf))

    get_pose3D(target, predicted_mhf, predicted_pf, output_dir)
    # img2video(video_path, output_dir)
    print('Generating demo successful!')
    
