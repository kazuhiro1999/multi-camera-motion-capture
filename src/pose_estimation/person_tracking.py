import numpy as np
from scipy.spatial.distance import cdist

class PersonTracker:
    def __init__(self, max_persons=2):
        self.previous_positions = {}  # ID: position
        self.last_id = 0
        self.max_persons = max_persons
        self.missing_threshold = 10  # フレーム数
        self.missing_counters = {}  # ID: missing_frames
    
    def get_person_position(self, landmarks):
        """ランドマークから人物の代表位置（上半身の重心）を計算"""
        # 上半身の主要なランドマーク（肩、腰、首など）の平均位置を使用
        upper_body_indices = [11, 12, 23, 24]  # 左右の肩と腰のインデックス
        positions = np.array([[landmarks[i].x, landmarks[i].y] for i in upper_body_indices])
        return np.mean(positions, axis=0)
    
    def assign_ids(self, current_landmarks):
        """現在のランドマークデータから安定したIDを割り当て"""
        if not current_landmarks:
            # 検出された人物がいない場合
            self.missing_counters = {k: v + 1 for k, v in self.missing_counters.items()}
            return {}
        
        current_positions = [self.get_person_position(lm) for lm in current_landmarks]
        
        # 前フレームの位置データがない場合は新規にIDを割り当て
        if not self.previous_positions:
            assignments = {i: i for i in range(len(current_positions))}
            self.previous_positions = {i: pos for i, pos in enumerate(current_positions)}
            self.missing_counters = {i: 0 for i in range(len(current_positions))}
            return assignments
        
        # 前フレームと現在のフレームの位置での距離行列を計算
        prev_pos = np.array(list(self.previous_positions.values()))
        curr_pos = np.array(current_positions)
        distance_matrix = cdist(prev_pos, curr_pos)
        
        # ハンガリアンアルゴリズムで最適な割り当てを計算
        from scipy.optimize import linear_sum_assignment
        prev_indices, curr_indices = linear_sum_assignment(distance_matrix)
        
        # 割り当ての結果をIDにマッピング
        assignments = {}
        prev_ids = list(self.previous_positions.keys())
        
        # 既存のIDを割り当て
        for prev_idx, curr_idx in zip(prev_indices, curr_indices):
            if distance_matrix[prev_idx, curr_idx] < 0.3:  # 距離の閾値
                prev_id = prev_ids[prev_idx]
                assignments[curr_idx] = prev_id
                self.missing_counters[prev_id] = 0
        
        # 新しい検出に新規IDを割り当て
        for i in range(len(current_positions)):
            if i not in assignments:
                while self.last_id in self.previous_positions:
                    self.last_id += 1
                assignments[i] = self.last_id
                self.missing_counters[self.last_id] = 0
                self.last_id += 1
        
        # 位置データを更新
        self.previous_positions = {assignments[i]: pos for i, pos in enumerate(current_positions)}
        
        # 長時間検出されていないIDを削除
        for id in list(self.missing_counters.keys()):
            if self.missing_counters[id] > self.missing_threshold:
                del self.missing_counters[id]
                if id in self.previous_positions:
                    del self.previous_positions[id]
        
        return assignments