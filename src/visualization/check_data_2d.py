import numpy as np
import pickle

# データの読み込み
with open('temp/pose2d_keito_cFL.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['metadata'])

# 特定の人物の全フレームのデータにアクセス
person_id = "1"
pose2d = data['pose'][person_id]['pose2d']  # (n_frames, 33, 4)
pose3d = data['pose'][person_id]['pose3d']  # (n_frames, 33, 3)
frame_ids = data['pose'][person_id]['frame_ids']

print(pose2d.shape)
print(pose3d.shape)

# 欠損フレームを除外した有効なデータの取得
valid_pose2d = pose2d[~np.isnan(pose2d).any(axis=(1,2))]