import numpy as np
import cv2
import pickle
import mediapipe as mp
from tqdm import tqdm


def visualize_pose2d_with_video(video_path, pkl_path, output_video_path):

    # MediaPipe関連の初期化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # データの読み込み
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    person_id = 0
    pose2d = data['pose'][person_id]['pose2d']  # (n_frames, 33, 4)
    frame_ids = data['pose'][person_id]['frame_ids']

    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 動画出力設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # フレームごとの処理
    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # フレームに対応するランドマークが存在する場合
        if frame_idx in frame_ids:
            landmarks_2d = pose2d[frame_idx]
            
            # ランドマークを描画
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if (
                    not np.isnan(landmarks_2d[start_idx, 0]) and
                    not np.isnan(landmarks_2d[end_idx, 0])
                ):
                    start_point = tuple(landmarks_2d[start_idx, :2].astype(int))
                    end_point = tuple(landmarks_2d[end_idx, :2].astype(int))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

            for landmark in landmarks_2d:
                if not np.isnan(landmark[0]):
                    center = tuple(landmark[:2].astype(int))
                    if landmark[2] >= 0.5:
                        cv2.circle(frame, center, 3, (0, 255, 0), -1)
                    else:
                        cv2.circle(frame, center, 3, (0, 0, 255), -1)

        # フレームを出力
        out.write(frame)

        # 動画のリソースを解放
    cap.release()
    out.release()

    print("Landmark overlay video has been saved to:", output_video_path)