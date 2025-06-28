import numpy as np
import pickle
from collections import defaultdict

class PoseDataCollector:
    def __init__(self):
        # 人物IDごとの2D/3D座標データを保持
        self.pose_data = defaultdict(lambda: {
            'pose2d': [],  # フレームごとの2D座標リスト
            'pose3d': [],  # フレームごとの3D座標リスト
            'frame_ids': []  # データが存在するフレームIDリスト
        })
        self.frame_counter = 0
        
    def add_frame_data(self, results, image_shape=(640,360)):
        """フレームごとの姿勢データを追加"""
        if results.pose_landmarks:
            person_id = 0
            landmarks = results.pose_landmarks.landmark
            
            # 2D座標の取得 (x, y, visibility, presence)
            w, h = image_shape
            pose2d = np.array([[landmark.x * w, landmark.y * h, landmark.visibility, landmark.presence] for landmark in landmarks])
                
            # 3D座標の取得 (x, y, z)
            # MediaPipeの座標系を調整 (y, zを入れ替えて-をつける)
            if results.pose_world_landmarks:
                pose3d = np.array([
                    [landmark.x, -landmark.y, landmark.z] 
                    for landmark in results.pose_world_landmarks.landmark
                ])
            else:
                pose3d = None
                
            # データの追加
            self.pose_data[person_id]['pose2d'].append(pose2d)
            self.pose_data[person_id]['pose3d'].append(pose3d)
            self.pose_data[person_id]['frame_ids'].append(self.frame_counter)    
                
        self.frame_counter += 1
    
    def save_to_file(self, filename: str):
        """データを整形してpklファイルとして保存"""
        formatted_data = {
            'metadata': {
                'total_frames': self.frame_counter,
                'num_persons': len(self.pose_data)
            },
            'pose': {}
        }
        
        # 各人物のデータを整形
        for person_id, person_data in self.pose_data.items():
            # フレーム数 x 関節数 x 2/3 の配列を作成
            n_frames = self.frame_counter
            n_joints = 33  # MediaPipeの関節数
            
            # 欠損フレームを考慮して初期化
            pose2d = np.full((n_frames, n_joints, 4), np.nan)
            pose3d = np.full((n_frames, n_joints, 3), np.nan)
            
            # データの格納
            for frame_idx, (pose2d_frame, pose3d_frame, orig_frame_id) in enumerate(
                zip(person_data['pose2d'], person_data['pose3d'], person_data['frame_ids'])
            ):
                pose2d[orig_frame_id] = pose2d_frame
                if pose3d_frame is not None:
                    pose3d[orig_frame_id] = pose3d_frame
            
            formatted_data['pose'][person_id] = {
                'pose2d': pose2d,  # shape: (n_frames, n_joints, 4)
                'pose3d': pose3d,  # shape: (n_frames, n_joints, 3)
                'frame_ids': np.array(person_data['frame_ids'])  # 実際にデータが存在するフレームIDのリスト
            }
        
        # ファイルに保存
        with open(filename, 'wb') as f:
            pickle.dump(formatted_data, f)
        
        print(f"[Info] Saved data for {len(self.pose_data)} persons over {self.frame_counter} frames to {filename}")

    

class PoseDataCollectorMultiPerson:
    def __init__(self):
        # 人物IDごとの2D/3D座標データを保持
        self.pose_data = defaultdict(lambda: {
            'pose2d': [],  # フレームごとの2D座標リスト
            'pose3d': [],  # フレームごとの3D座標リスト
            'frame_ids': []  # データが存在するフレームIDリスト
        })
        self.frame_counter = 0
        
    def add_frame_data(self, detection_result, id_assignments, image_shape=(640,360)):
        """フレームごとの姿勢データを追加"""
        if detection_result.pose_landmarks:
            for i, landmarks in enumerate(detection_result.pose_landmarks):
                person_id = id_assignments.get(i, i)
                
                # 2D座標の取得 (x, y, visibility, presence)
                w, h = image_shape
                pose2d = np.array([[landmark.x * w, landmark.y * h, landmark.visibility, landmark.presence] for landmark in landmarks])
                
                # 3D座標の取得 (x, y, z)
                # MediaPipeの座標系を調整 (y, zを入れ替えて-をつける)
                if detection_result.pose_world_landmarks and i < len(detection_result.pose_world_landmarks):
                    pose3d = np.array([
                        [landmark.x, -landmark.y, landmark.z] 
                        for landmark in detection_result.pose_world_landmarks[i]
                    ])
                else:
                    pose3d = None
                
                # データの追加
                self.pose_data[person_id]['pose2d'].append(pose2d)
                self.pose_data[person_id]['pose3d'].append(pose3d)
                self.pose_data[person_id]['frame_ids'].append(self.frame_counter)
        
        self.frame_counter += 1
    
    def save_to_file(self, filename: str):
        """データを整形してpklファイルとして保存"""
        formatted_data = {
            'metadata': {
                'total_frames': self.frame_counter,
                'num_persons': len(self.pose_data)
            },
            'pose': {}
        }
        
        # 各人物のデータを整形
        for person_id, person_data in self.pose_data.items():
            # フレーム数 x 関節数 x 2/3 の配列を作成
            n_frames = self.frame_counter
            n_joints = 33  # MediaPipeの関節数
            
            # 欠損フレームを考慮して初期化
            pose2d = np.full((n_frames, n_joints, 4), np.nan)
            pose3d = np.full((n_frames, n_joints, 3), np.nan)
            
            # データの格納
            for frame_idx, (pose2d_frame, pose3d_frame, orig_frame_id) in enumerate(
                zip(person_data['pose2d'], person_data['pose3d'], person_data['frame_ids'])
            ):
                pose2d[orig_frame_id] = pose2d_frame
                if pose3d_frame is not None:
                    pose3d[orig_frame_id] = pose3d_frame
            
            formatted_data['pose'][person_id] = {
                'pose2d': pose2d,  # shape: (n_frames, n_joints, 4)
                'pose3d': pose3d,  # shape: (n_frames, n_joints, 3)
                'frame_ids': np.array(person_data['frame_ids'])  # 実際にデータが存在するフレームIDのリスト
            }
        
        # ファイルに保存
        with open(filename, 'wb') as f:
            pickle.dump(formatted_data, f)
        
        print(f"Saved data for {len(self.pose_data)} persons over {self.frame_counter} frames to {filename}")