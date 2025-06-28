import json
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

from src.motion.utils import calculate_calibration_data
from src.postprocessing.fix_pose3d import calculate_projection_matrix, extract_valid_2d_data, optimize_with_bone_length
from src.postprocessing.one_euro_filter import OneEuroFilter
from src.postprocessing.utils import KEYPOINTS_DICT


def refine_pose3d(videos, pose3d_pkl_path, output_path):

    if not os.path.exists(pose3d_pkl_path):
        print(f"[Error] file does not exists: {pose3d_pkl_path}")
        return

    # 3d keypoints
    with open(pose3d_pkl_path, 'rb') as p:
        pose3d = pickle.load(p)

    # 2d keypoints and camera parameters
    all_2d_data = []
    proj_matrices = []

    for video in videos:
        pkl_file_path = video.get('pose2d_output_path')
        camera_setting_path = video.get('camera_setting_path')

        if not os.path.exists(pkl_file_path) or not os.path.exists(camera_setting_path):
            print(f"[Error] file does not exists. please check pose2d and camera settings.")
            return

        # 2Dデータ読み込み
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            all_2d_data.append(data['pose'][0])

        # カメラ設定読み込み
        with open(camera_setting_path, 'r') as f:
            camera_params = json.load(f)
            intrinsic = camera_params.get('intrinsic_matrix')
            extrinsic = camera_params.get('extrinsic_matrix')
            proj_matrices.append(calculate_projection_matrix(intrinsic, extrinsic))

    # estimate skeleton (bone length)
    calbration_data = calculate_calibration_data(pose3d['pose3d'])

    # fix 3d position 
    for frame_i, keypoints3d in tqdm(enumerate(pose3d['pose3d']), file=sys.stdout):

        if keypoints3d is None:
            # 欠損データ -> フレーム補間
            continue

        # fix left arm
        l_shoulder_pos = keypoints3d[KEYPOINTS_DICT['left_shoulder']]
        l_elbow_pos = keypoints3d[KEYPOINTS_DICT['left_elbow']]
        l_wrist_pos = keypoints3d[KEYPOINTS_DICT['left_wrist']]

        # optimize elbow position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'left_elbow', proj_matrices)
        l_elbow_pos_optimized = optimize_with_bone_length(l_elbow_pos, points2d, used_proj_matrices, l_shoulder_pos, calbration_data.upperarm_length)

        # optimize wrist position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'left_wrist', proj_matrices)
        l_wrist_pos_optimized = optimize_with_bone_length(l_wrist_pos, points2d, used_proj_matrices, l_elbow_pos_optimized, calbration_data.lowerarm_length)


        # fix right arm
        r_shoulder_pos = keypoints3d[KEYPOINTS_DICT['right_shoulder']]
        r_elbow_pos = keypoints3d[KEYPOINTS_DICT['right_elbow']]
        r_wrist_pos = keypoints3d[KEYPOINTS_DICT['right_wrist']]

        # optimize elbow position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'right_elbow', proj_matrices)
        r_elbow_pos_optimized = optimize_with_bone_length(r_elbow_pos, points2d, used_proj_matrices, r_shoulder_pos, calbration_data.upperarm_length)

        # optimize wrist position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'right_wrist', proj_matrices)
        r_wrist_pos_optimized = optimize_with_bone_length(r_wrist_pos, points2d, used_proj_matrices, r_elbow_pos_optimized, calbration_data.lowerarm_length)


        # fix left leg
        l_hip_pos = keypoints3d[KEYPOINTS_DICT['left_hip']]
        l_knee_pos = keypoints3d[KEYPOINTS_DICT['left_knee']]
        l_ankle_pos = keypoints3d[KEYPOINTS_DICT['left_ankle']]

        # optimize knee position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'left_knee', proj_matrices)
        l_knee_pos_optimized = optimize_with_bone_length(l_knee_pos, points2d, used_proj_matrices, l_hip_pos, calbration_data.upperleg_length)

        # optimize ankle position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'left_ankle', proj_matrices)
        l_ankle_pos_optimized = optimize_with_bone_length(l_ankle_pos, points2d, used_proj_matrices, l_knee_pos_optimized, calbration_data.lowerleg_length)

        # fix right leg
        r_hip_pos = keypoints3d[KEYPOINTS_DICT['right_hip']]
        r_knee_pos = keypoints3d[KEYPOINTS_DICT['right_knee']]
        r_ankle_pos = keypoints3d[KEYPOINTS_DICT['right_ankle']]

        # optimize knee position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'right_knee', proj_matrices)
        r_knee_pos_optimized = optimize_with_bone_length(r_knee_pos, points2d, used_proj_matrices, r_hip_pos, calbration_data.upperleg_length)

        # optimize ankle position
        points2d, used_proj_matrices = extract_valid_2d_data(all_2d_data, frame_i, 'right_ankle', proj_matrices)
        r_ankle_pos_optimized = optimize_with_bone_length(r_ankle_pos, points2d, used_proj_matrices, r_knee_pos_optimized, calbration_data.lowerleg_length)


        # 元データの修正
        keypoints3d[KEYPOINTS_DICT['left_elbow']] = l_elbow_pos_optimized
        keypoints3d[KEYPOINTS_DICT['left_wrist']] = l_wrist_pos_optimized
        keypoints3d[KEYPOINTS_DICT['right_elbow']] = r_elbow_pos_optimized
        keypoints3d[KEYPOINTS_DICT['right_wrist']] = r_wrist_pos_optimized
        keypoints3d[KEYPOINTS_DICT['left_knee']] = l_knee_pos_optimized
        keypoints3d[KEYPOINTS_DICT['left_ankle']] = l_ankle_pos_optimized
        keypoints3d[KEYPOINTS_DICT['right_knee']] = r_knee_pos_optimized
        keypoints3d[KEYPOINTS_DICT['right_ankle']] = r_ankle_pos_optimized

    # 修正版データの保存
    with open(output_path, 'wb') as f:
        pickle.dump(pose3d, f)

    print(f"[Info] refine completed: {output_path}")


def filter_pose3d(pose3d_pkl_path, output_path, frame_rate=60.0, verbose=1):
    if not os.path.exists(pose3d_pkl_path):
        print(f"[Error] file does not exists: {pose3d_pkl_path}")
        return

    # 3d keypoints
    with open(pose3d_pkl_path, 'rb') as p:
        pose3d = pickle.load(p)

    filter = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=2.0)
    all_3d_poses = []
    logs = { "None": [], "NaN": [] }

    for frame_i, keypoints3d in tqdm(enumerate(pose3d['pose3d']), file=sys.stdout):
        t = 1000 * frame_i / frame_rate

        if keypoints3d is None:
            logs['None'].append(frame_i)
            continue

        if np.any(np.isnan(keypoints3d)):
            logs['NaN'].append(frame_i)
            continue
        
        filtered_keypoints3d = filter.apply(t, keypoints3d) 
        all_3d_poses.append(filtered_keypoints3d)


    pose3d['pose3d'] = all_3d_poses

    if verbose > 1:
        print(f"value is None: {logs['None']}")
        print(f"value is NaN: {logs['NaN']}")

    # 修正版データの保存
    with open(output_path, 'wb') as f:
        pickle.dump(pose3d, f)

    print(f"[Info] filtering completed: {output_path}")

    return logs
