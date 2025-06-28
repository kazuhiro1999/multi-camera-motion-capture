import sys
import cv2
import mediapipe as mp
from tqdm import tqdm

from src.pose_estimation.pose_collector import PoseDataCollector

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def process_video(video_path, output_pkl_path, model_complexity=2):
    ''' 
    2D姿勢推定 => temp/pose2d_*.pkl    
    '''
    
    ## 動画読み込み
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        pose_collector = PoseDataCollector()
        print(f"[Info] Processing video: {video_path}")

        for _ in tqdm(range(frame_count), file=sys.stdout):
            ret, frame = cap.read()
            if not ret:
                break

            # RGB 変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 姿勢推定
            results = pose.process(frame_rgb)

            # データを追加
            pose_collector.add_frame_data(results, image_shape=(frame_width, frame_height))

        cap.release()
        pose_collector.save_to_file(output_pkl_path)


def draw_2d_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    connections = mp.solutions.pose.POSE_CONNECTIONS
    
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * w), 
                      int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), 
                    int(landmarks[end_idx].y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
    for landmark in landmarks:
        point = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(frame, point, 5, (255, 0, 0), -1)


def draw_3d_landmarks(ax, landmarks):
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    
    x = [landmark.x for landmark in landmarks]
    y = [landmark.z for landmark in landmarks]  # MediaPipeの座標系を調整
    z = [-landmark.y for landmark in landmarks]  # MediaPipeの座標系を調整
    
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot([x[start_idx], x[end_idx]],
                [y[start_idx], y[end_idx]],
                [z[start_idx], z[end_idx]], 'b-')
    
    ax.scatter(x, y, z, c='r', marker='o')


def draw_2d_landmarks_with_id(frame, landmarks, person_id=-1):
    h, w = frame.shape[:2]
    connections = mp.solutions.pose.POSE_CONNECTIONS
    
    # 人物ごとに異なる色を使用
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 緑、赤、青
    color = colors[person_id % len(colors)]
    
    # ランドマークとつながりの描画
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * w), 
                      int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), 
                    int(landmarks[end_idx].y * h))
        cv2.line(frame, start_point, end_point, color, 2)
        
    for landmark in landmarks:
        point = (int(landmark.x * w), int(landmark.y * h))
        if landmark.visibility > 0.5:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
        else:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
    
    # IDの表示（頭の上に表示）
    nose_point = (int(landmarks[0].x * w), int(landmarks[0].y * h))  # 鼻のランドマーク
    id_position = (nose_point[0], nose_point[1] - 30)  # 頭上に表示位置を設定
    
    # IDのテキスト描画（背景付き）
    text = f"ID: {person_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # テキストの背景を描画
    cv2.rectangle(frame, 
                 (id_position[0] - 5, id_position[1] - text_size[1] - 5),
                 (id_position[0] + text_size[0] + 5, id_position[1] + 5),
                 (255, 255, 255),
                 -1)
    
    # テキストを描画
    cv2.putText(frame, 
                text,
                id_position,
                font,
                font_scale,
                color,
                thickness)
