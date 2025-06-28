import sys
import pickle
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

POSE_CONNECTIONS = [
    (11, 12), (12, 24), (24, 23), (23, 11), 
    (11, 13), (13, 15),  
    (12, 14), (14, 16), 
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

def draw_3d_pose(ax, keypoints3d, color='green', linecolor='black'):
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)

    for x, y, z in keypoints3d:
        ax.scatter(x, z, y, c=color, marker='o')

    # 接続線を描画
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if (
            not np.isnan(keypoints3d[start_idx]).any() and
            not np.isnan(keypoints3d[end_idx]).any()
        ):
            start_point = keypoints3d[start_idx]
            end_point = keypoints3d[end_idx]
            ax.plot(
                [start_point[0], end_point[0]],  # x座標
                [start_point[2], end_point[2]],  # z座標
                [start_point[1], end_point[1]],  # y座標
                c=linecolor  # 線の色
            )

def visualize_pose3d(pkl_path, output_video_path, fps=60):

    # データの読み込み
    with open(pkl_path, 'rb') as p:
        data = pickle.load(p)

    frame_count = len(data['pose3d'])

    # 3Dプロット用の設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 動画保存設定
    writer = FFMpegWriter(fps=fps, metadata=dict(title="3D Pose Animation", artist="Matplotlib"))

    # 動画保存の開始
    with writer.saving(fig, output_video_path, dpi=100):
        for i, pose3d in tqdm(enumerate(data['pose3d']), file=sys.stdout):
            if pose3d is not None:
                pose3d = np.nan_to_num(pose3d, nan=0)
                draw_3d_pose(ax, pose3d, color='black')
                writer.grab_frame()  # フレームを保存
            else:
                pass
