import os
import sys
import json
import pickle
from tqdm import tqdm

from src.pose_estimation.pose3d import *


# 3D姿勢推定を行うプログラム
def process_3d(videos, output_filepath, verbose=1):
    '''
    3D姿勢推定 => pose3d.pkl
    '''

    # 2Dデータとカメラパラメータを読み込む
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


    # 全フレームでの処理
    num_frames = max(len(data['pose2d']) for data in all_2d_data)  # 最大フレーム数を推定
    all_3d_poses = []
    failed_frames = []

    for frame_idx in tqdm(range(num_frames), file=sys.stdout):
        keypoints2d_list = []
        used_proj_matrices = []

        # 有効な視点の2Dデータと射影行列を収集
        for view_idx, data in enumerate(all_2d_data):
            if frame_idx in data['frame_ids']:  # この視点でフレームが有効か確認
                frame_data_idx = np.where(data['frame_ids'] == frame_idx)[0][0]
                keypoints2d = data['pose2d'][frame_data_idx]
                keypoints2d_list.append(keypoints2d)
                used_proj_matrices.append(proj_matrices[view_idx])

        # 3D姿勢推定
        if len(keypoints2d_list) >= 2:  # 有効な視点が2つ以上必要
            keypoints3d, failed_joints = recover_pose_3d(used_proj_matrices, keypoints2d_list)

            if verbose > 1 and len(failed_joints) > 0:
                print(f"Failed during triangulation: frame {frame_idx} {failed_joints}")

        else:
            keypoints3d = None  # 再構成できない場合はNoneを設定

        all_3d_poses.append(keypoints3d)

    # 結果を保存
    result_data = {
        'proj_matrices': proj_matrices,
        'pose3d': all_3d_poses,
    }
    with open(output_filepath, 'wb') as f:
        pickle.dump(result_data, f)

    print(f"[Info] 3D pose estimation completed and saved to {output_filepath}")


def draw_3d_pose(ax, keypoints3d, color='r'):
    ax.cla()
    ax.set_xlim3d(-1.5, 1.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_zlim3d(0, 3)
    
    for x, y, z in keypoints3d:
        ax.scatter(x, z, y, c=color, marker='o')


if __name__ == '__main__':
    # 実行例
    pkl_files = [
        'temp/pose2d_keito_cFL.pkl', 
        'temp/pose2d_keito_cFR.pkl', 
        'temp/pose2d_keito_cBL.pkl', 
        'temp/pose2d_keito_cBR.pkl'
    ]
    camera_configs = [
        'setting/camera_setting_cFL.json', 
        'setting/camera_setting_cFR.json', 
        'setting/camera_setting_cBL.json', 
        'setting/camera_setting_cBR.json'
    ]
    output_file = 'output/pose3d_keito.pkl'

    process_3d(pkl_files, camera_configs, output_file)
