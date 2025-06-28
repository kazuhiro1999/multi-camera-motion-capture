import sys
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from src.postprocessing.one_euro_filter import OneEuroFilter  # 既存のOneEuroFilterクラス

pkl_path = "reference_ws2023.mot"
output_video_path = "reference_ws2023_filtered.mp4"

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

# データの読み込み
with open(pkl_path, 'rb') as p:
    data = pickle.load(p)

frame_count = len(data['pose3d'])

# OneEuroFilter の初期化
filter = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=2.0)

# 3Dプロット用の設定
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 動画保存設定
fps = 60  # フレームレート
writer = FFMpegWriter(fps=fps, metadata=dict(title="3D Pose Animation", artist="Matplotlib"))

# 動画保存の開始
with writer.saving(fig, output_video_path, dpi=100):
    for i, pose3d in tqdm(enumerate(data['pose3d']), file=sys.stdout):
        t = 1000 * i / fps  # milliseconds
        if pose3d is not None:
            pose3d = np.nan_to_num(pose3d, nan=0)
            #pose3d = filter.apply(t, pose3d)
            draw_3d_pose(ax, pose3d, color='black')
            writer.grab_frame()  # フレームを保存
        else:
            print("pose is None")
