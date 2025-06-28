import os
import sys
import json
import numpy as np
import pickle
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import least_squares, minimize

from src.motion.utils import calculate_calibration_data
from src.postprocessing.utils import *


def refine_pose3d(videos, pose3d_pkl_path, output_path):

    if not os.path.exists(pose3d_pkl_path):
        print(f"file does not exists: {pose3d_pkl_path}")
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
            print(f"file does not exists. please check pose2d and camera settings.")
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

    print(f"refine completed: {output_path}")


def optimize_with_bone_length(point3d, points2d, proj_matrices, parent_pos, length, method='least_squares', verbose=1):

    def objective_function(point):
        point = np.array(point)
        reprojection_err = calculate_reprojection_error(point, points2d, proj_matrices)
        length_err = calculate_length_error(point, parent_pos, length)

        # NaNやInfをチェック
        if np.isnan(reprojection_err).any() or np.isinf(reprojection_err).any():
            if verbose > 1:
                print("NaN or Inf found in reprojection_err")
            return np.inf  # 無限大を返して最適化を停止する

        if np.isnan(length_err).any() or np.isinf(length_err).any():
            if verbose > 1:
                print("NaN or Inf found in length_err")
            return np.inf  # 無限大を返して最適化を停止する
        
        # ToDo: weight
        return reprojection_err + 10 * length_err

    try:
        result = minimize(objective_function, point3d, method='L-BFGS-B')
        #result = least_squares(objective_function, point3d, loss='huber', method='trf')
        if result.success:
            optimized_point3d = result.x
            #print(f"Optimization succeeded. Optimized position: {optimized_point3d}\nInitial position: {point3d}")
        else:
            if verbose > 1:
                print(f"Optimization failed: {result.message}")
            optimized_point3d = point3d  # 初期値を返す
        return optimized_point3d
    except Exception as e:
        if verbose > 1:
            print(f"Optimization failed: {e}")
        return point3d  # 初期値を返す

# 再投影
def reprojection(point3d, proj_matrix):
    assert point3d.shape == (3,)
    assert proj_matrix.shape == (3, 4)
    point3d_homogeneous = np.append(point3d, 1)  # shape (4,)
    point2d_homogeneous = proj_matrix @ point3d_homogeneous  # shape (3,)
    point2d = point2d_homogeneous[:2] / point2d_homogeneous[2]
    return point2d

# 再投影誤差
def calculate_reprojection_error(point3d, points2d, proj_matrices, sigma=5.0):
    # ToDo: sigma仮置き visibilityの活用
    error = 0
    for point2d, proj_matrix in zip(points2d, proj_matrices):
        reprojected = reprojection(point3d, proj_matrix)       
        distance = np.linalg.norm(reprojected - point2d[:2])  # 距離（再投影誤差）        
        visibility = point2d[2]  # visibility (0: 見えない, 1: 完全に見える) 
        error += (norm.pdf(0, loc=0, scale=sigma) - norm.pdf(distance, loc=0, scale=sigma)) * distance
    return error

def calculate_length_error(point3d, parent_pos, length):
    current_length = np.linalg.norm(point3d - parent_pos)
    return (current_length - length) ** 2

# カメラの射影行列を計算
def calculate_projection_matrix(intrinsic, extrinsic):
    return np.dot(intrinsic, extrinsic)

# 有効な視点の2Dデータと射影行列を収集
def extract_valid_2d_data(all_2d_data, frame_i, key, proj_matrices):
    points2d = []
    used_proj_matrices = []
    for view_idx, pose2d in enumerate(all_2d_data):
        if frame_i in pose2d['frame_ids']:
            frame_data_idx = np.where(pose2d['frame_ids'] == frame_i)[0][0]
            point2d = pose2d['pose2d'][frame_data_idx, KEYPOINTS_DICT[key]]
            points2d.append(point2d)
            used_proj_matrices.append(proj_matrices[view_idx])
    return points2d, used_proj_matrices