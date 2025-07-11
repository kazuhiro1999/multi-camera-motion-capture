# Unity用データ変換

import sys
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from src.postprocessing.utils import *
from src.motion.motionfile import *


# 関節の回転を計算するための関数
def pose_to_body_transforms(pose):
    nose_pos = pose[KEYPOINTS_DICT['nose']]
    l_ear_pos = pose[KEYPOINTS_DICT['left_ear']]
    r_ear_pos = pose[KEYPOINTS_DICT['right_ear']]
    l_eye_pos = pose[KEYPOINTS_DICT['left_eye']]
    r_eye_pos = pose[KEYPOINTS_DICT['right_eye']]
    l_mouth_pos = pose[KEYPOINTS_DICT['left_mouth']]    
    r_mouth_pos = pose[KEYPOINTS_DICT['right_mouth']]
    
    l_shoulder_pos = pose[KEYPOINTS_DICT['left_shoulder']]
    l_elbow_pos = pose[KEYPOINTS_DICT['left_elbow']]
    l_wrist_pos = pose[KEYPOINTS_DICT['left_wrist']]
    l_inner_hand_pos = pose[KEYPOINTS_DICT['left_inner_hand']]
    l_outer_hand_pos = pose[KEYPOINTS_DICT['left_outer_hand']]   
    l_hand_tip_pos = pose[KEYPOINTS_DICT['left_hand_tip']]
    
    r_shoulder_pos = pose[KEYPOINTS_DICT['right_shoulder']]
    r_elbow_pos = pose[KEYPOINTS_DICT['right_elbow']]
    r_wrist_pos = pose[KEYPOINTS_DICT['right_wrist']]
    r_inner_hand_pos = pose[KEYPOINTS_DICT['right_inner_hand']]
    r_outer_hand_pos = pose[KEYPOINTS_DICT['right_outer_hand']]
    r_hand_tip_pos = pose[KEYPOINTS_DICT['right_hand_tip']]

    l_hip_pos = pose[KEYPOINTS_DICT['left_hip']]
    l_knee_pos = pose[KEYPOINTS_DICT['left_knee']]
    l_ankle_pos = pose[KEYPOINTS_DICT['left_ankle']]
    l_heel_pos = pose[KEYPOINTS_DICT['left_heel']]
    l_toe_pos = pose[KEYPOINTS_DICT['left_toe']]
    
    r_hip_pos = pose[KEYPOINTS_DICT['right_hip']]
    r_knee_pos = pose[KEYPOINTS_DICT['right_knee']]
    r_ankle_pos = pose[KEYPOINTS_DICT['right_ankle']]
    r_heel_pos = pose[KEYPOINTS_DICT['right_heel']]
    r_toe_pos = pose[KEYPOINTS_DICT['right_toe']]    
    
    m_shoulder_pos = (l_shoulder_pos + r_shoulder_pos) / 2
    m_hip_pos = (l_hip_pos + r_hip_pos) / 2
    body_core_vector = m_shoulder_pos - m_hip_pos
    body_core_length =  np.linalg.norm(body_core_vector)
    body_core_direction = body_core_vector / (body_core_length + 1e-5)
    hip_pos = m_hip_pos + body_core_direction * body_core_length * 0.1
    neck_pos = m_shoulder_pos + body_core_direction * body_core_length * 0.1
    head_center_pos = (l_ear_pos + r_ear_pos) / 2 
    head_pos = (2 * neck_pos + 2 * head_center_pos) / 4
    chest_pos = (hip_pos + neck_pos) / 2 
    mouth_pos = (l_mouth_pos + r_mouth_pos) / 2   
    
    body_transform = defaultdict(dict)

    # 腰    
    hip_forward = np.cross(r_hip_pos - neck_pos, l_hip_pos - neck_pos)
    hip_right = r_hip_pos - l_hip_pos
    hip_up = np.cross(hip_right, hip_forward)
    hip_rotation_matrix = rotation_matrix_from_vectors(hip_forward, hip_up)
    body_transform['Hips']['position'] = hip_pos.tolist()
    body_transform['Hips']['rotation'] = matrix_to_quaternion(hip_rotation_matrix)

    # 胸   
    body_forward = np.cross(l_shoulder_pos - hip_pos, r_shoulder_pos - hip_pos)
    body_right = r_shoulder_pos - l_shoulder_pos
    body_up = np.cross(body_right, body_forward)
    body_rotation_matrix = rotation_matrix_from_vectors(body_forward, body_up)
    body_transform['Chest']['position'] = chest_pos.tolist()
    body_transform['Chest']['rotation'] = matrix_to_quaternion(body_rotation_matrix)

    # 首
    neck_up = (head_pos - neck_pos)
    neck_forward = body_forward  # 上半身の前方向を使用
    neck_rotation_matrix = rotation_matrix_from_vectors(neck_forward, neck_up)
    body_transform['Neck']['position'] = neck_pos.tolist()
    body_transform['Neck']['rotation'] = matrix_to_quaternion(neck_rotation_matrix)

    # 頭
    head_forward = np.cross(l_eye_pos - mouth_pos, r_eye_pos - mouth_pos)
    head_right = r_ear_pos - l_ear_pos
    head_up = np.cross(head_right, head_forward)
    head_rotation_matrix = rotation_matrix_from_vectors(head_forward, head_up)
    body_transform['Head']['position'] = head_pos.tolist()
    body_transform['Head']['rotation'] = matrix_to_quaternion(head_rotation_matrix)

    # ... 左腕の回転を計算 ...    
    # 左上腕
    l_upperarm_right = -(l_elbow_pos - l_shoulder_pos)
    # 腕が伸びているときは捻りが計算できないため、末端を使用
    l_elbow_angle = np.degrees(np.arccos(np.dot((l_shoulder_pos - l_elbow_pos), (l_wrist_pos - l_elbow_pos)) / 
                        (np.linalg.norm(l_shoulder_pos - l_elbow_pos) * np.linalg.norm(l_wrist_pos - l_elbow_pos))))
    if (l_elbow_angle < 160):
        l_upperarm_up = np.cross(l_shoulder_pos - l_elbow_pos, l_wrist_pos - l_elbow_pos)
    else:
        l_hand_forward = l_inner_hand_pos - l_outer_hand_pos
        l_upperarm_up = np.cross(l_upperarm_right, l_hand_forward)

    l_upperarm_up = np.cross(l_shoulder_pos - l_elbow_pos, l_wrist_pos - l_elbow_pos)
    l_upperarm_forward = np.cross(l_upperarm_up, l_upperarm_right)
    l_upperarm_up = np.cross(l_upperarm_right, l_upperarm_forward)
    l_upperarm_rotation_matrix = rotation_matrix_from_vectors(l_upperarm_forward, l_upperarm_up)
    body_transform['LeftUpperArm']['position'] = l_shoulder_pos.tolist()
    body_transform['LeftUpperArm']['rotation'] = matrix_to_quaternion(l_upperarm_rotation_matrix)

    # 左下腕   
    l_lowerarm_right = -(l_wrist_pos - l_elbow_pos)
    l_lowerarm_forward = l_inner_hand_pos - l_outer_hand_pos
    l_lowerarm_up = np.cross(l_lowerarm_right, l_lowerarm_forward)
    l_lowerarm_forward = np.cross(l_lowerarm_up, l_lowerarm_right)
    l_lowerarm_rotation_matrix = rotation_matrix_from_vectors(l_lowerarm_forward, l_lowerarm_up)
    body_transform['LeftLowerArm']['position'] = l_elbow_pos.tolist()
    body_transform['LeftLowerArm']['rotation'] = matrix_to_quaternion(l_lowerarm_rotation_matrix)
    
    # 左手  
    l_hand_right = -(l_hand_tip_pos - l_wrist_pos)
    l_hand_up = np.cross(l_inner_hand_pos - l_wrist_pos, l_outer_hand_pos - l_wrist_pos)    
    l_hand_forward = np.cross(l_hand_up, l_hand_right)
    l_hand_up = np.cross(l_hand_right, l_hand_forward)
    l_hand_rotation_matrix = rotation_matrix_from_vectors(l_hand_forward, l_hand_up)
    body_transform['LeftHand']['position'] = l_wrist_pos.tolist()
    body_transform['LeftHand']['rotation'] = matrix_to_quaternion(l_hand_rotation_matrix)

     # 右上腕
    r_upperarm_right = (r_elbow_pos - r_shoulder_pos)
    # 腕が伸びているときは捻りが計算できないため、末端を使用
    r_elbow_angle = np.degrees(np.arccos(np.dot((r_shoulder_pos - r_elbow_pos), (r_wrist_pos - r_elbow_pos)) / 
                        (np.linalg.norm(r_shoulder_pos - r_elbow_pos) * np.linalg.norm(r_wrist_pos - r_elbow_pos))))
    if (r_elbow_angle < 160):
        r_upperarm_up = np.cross(r_wrist_pos - r_elbow_pos, r_shoulder_pos - r_elbow_pos)
    else:
        r_hand_forward = r_inner_hand_pos - r_outer_hand_pos
        r_upperarm_up = np.cross(r_upperarm_right, r_hand_forward)

    r_upperarm_forward = np.cross(r_upperarm_up, r_upperarm_right)
    r_upperarm_up = np.cross(r_upperarm_right, r_upperarm_forward)
    r_upperarm_rotation_matrix = rotation_matrix_from_vectors(r_upperarm_forward, r_upperarm_up)
    body_transform['RightUpperArm']['position'] = r_shoulder_pos.tolist()
    body_transform['RightUpperArm']['rotation'] = matrix_to_quaternion(r_upperarm_rotation_matrix)

    # 右下腕
    r_lowerarm_right = (r_wrist_pos - r_elbow_pos)  
    r_lowerarm_forward = r_inner_hand_pos - r_outer_hand_pos
    r_lowerarm_up = np.cross(r_lowerarm_right, r_lowerarm_forward)
    r_lowerarm_forward = np.cross(r_lowerarm_up, r_lowerarm_right)
    r_lowerarm_rotation_matrix = rotation_matrix_from_vectors(r_lowerarm_forward, r_lowerarm_up)
    body_transform['RightLowerArm']['position'] = r_elbow_pos.tolist()
    body_transform['RightLowerArm']['rotation'] = matrix_to_quaternion(r_lowerarm_rotation_matrix)

    # 右手
    r_hand_right = (r_hand_tip_pos - r_wrist_pos) 
    r_hand_up = np.cross(r_outer_hand_pos - r_wrist_pos, r_inner_hand_pos - r_wrist_pos)
    r_hand_forward = np.cross(r_hand_up, r_hand_right)
    r_hand_up = np.cross(r_hand_right, r_hand_forward)
    r_hand_rotation_matrix = rotation_matrix_from_vectors(r_hand_forward, r_hand_up)
    body_transform['RightHand']['position'] = r_wrist_pos.tolist()
    body_transform['RightHand']['rotation'] = matrix_to_quaternion(r_hand_rotation_matrix)

    # 左上脚
    l_upperleg_up = -(l_knee_pos - l_hip_pos)
    l_knee_angle = np.degrees(np.arccos(np.dot((l_hip_pos - l_knee_pos), (l_ankle_pos - l_knee_pos)) / 
                        (np.linalg.norm(l_hip_pos - l_knee_pos) * np.linalg.norm(l_ankle_pos - l_knee_pos))))
    if (l_knee_angle < 160):
        l_upperleg_right = np.cross(l_hip_pos - l_knee_pos, l_ankle_pos - l_knee_pos)
    else:
        l_upperleg_right = np.cross(l_heel_pos - l_hip_pos, l_toe_pos - l_hip_pos)
    l_upperleg_forward = np.cross(l_upperleg_up, l_upperleg_right)
    l_upperleg_rotation_matrix = rotation_matrix_from_vectors(l_upperleg_forward, l_upperleg_up)
    body_transform['LeftUpperLeg']['position'] = l_hip_pos.tolist()
    body_transform['LeftUpperLeg']['rotation'] = matrix_to_quaternion(l_upperleg_rotation_matrix)

    # 左膝
    l_lowerleg_up = -(l_ankle_pos - l_knee_pos)
    l_lowerleg_right = np.cross(l_heel_pos - l_knee_pos, l_toe_pos - l_knee_pos)
    l_lowerleg_forward = np.cross(l_lowerleg_up, l_lowerleg_right)
    l_lowerleg_rotation_matrix = rotation_matrix_from_vectors(l_lowerleg_forward, l_lowerleg_up)
    body_transform['LeftLowerLeg']['position'] = l_knee_pos.tolist()
    body_transform['LeftLowerLeg']['rotation'] = matrix_to_quaternion(l_lowerleg_rotation_matrix)

    # 左足首
    l_foot_forward = l_toe_pos - l_heel_pos
    l_foot_right = np.cross(l_heel_pos - l_ankle_pos, l_toe_pos - l_ankle_pos)
    l_foot_up = np.cross(l_foot_right, l_foot_forward)
    l_foot_rotation_matrix = rotation_matrix_from_vectors(l_foot_forward, l_foot_up)
    body_transform['LeftFoot']['position'] = l_ankle_pos.tolist()
    body_transform['LeftFoot']['rotation'] = matrix_to_quaternion(l_foot_rotation_matrix)

    # 左つま先
    body_transform['LeftToes']['position'] = l_toe_pos.tolist()
    body_transform['LeftToes']['rotation'] = matrix_to_quaternion(l_foot_rotation_matrix)

    # 右上脚
    r_upperleg_up = -(r_knee_pos - r_hip_pos)
    r_knee_angle = np.degrees(np.arccos(np.dot((r_hip_pos - r_knee_pos), (r_ankle_pos - r_knee_pos)) / 
                        (np.linalg.norm(r_hip_pos - r_knee_pos) * np.linalg.norm(r_ankle_pos - r_knee_pos))))
    if (r_knee_angle < 160):
        r_upperleg_right = np.cross(r_hip_pos - r_knee_pos, r_ankle_pos - r_knee_pos)  
    else:
        r_upperleg_right = np.cross(r_heel_pos - r_hip_pos, r_toe_pos - r_hip_pos) 
    r_upperleg_forward = np.cross(r_upperleg_up, r_upperleg_right)
    r_upperleg_rotation_matrix = rotation_matrix_from_vectors(r_upperleg_forward, r_upperleg_up)
    body_transform['RightUpperLeg']['position'] = r_hip_pos.tolist()
    body_transform['RightUpperLeg']['rotation'] = matrix_to_quaternion(r_upperleg_rotation_matrix)

    # 右膝
    r_lowerleg_up = -(r_ankle_pos - r_knee_pos)
    r_lowerleg_right = np.cross(r_heel_pos - r_knee_pos, r_toe_pos - r_knee_pos)
    r_lowerleg_forward = np.cross(r_lowerleg_up, r_lowerleg_right)
    r_lowerleg_rotation_matrix = rotation_matrix_from_vectors(r_lowerleg_forward, r_lowerleg_up)
    body_transform['RightLowerLeg']['position'] = r_knee_pos.tolist()
    body_transform['RightLowerLeg']['rotation'] = matrix_to_quaternion(r_lowerleg_rotation_matrix)
    
    # 右足首
    r_foot_forward = r_toe_pos - r_heel_pos
    r_foot_right = np.cross(r_heel_pos - r_ankle_pos, r_toe_pos - r_ankle_pos)  
    r_foot_up = np.cross(r_foot_right, r_foot_forward)  
    r_foot_rotation_matrix = rotation_matrix_from_vectors(r_foot_forward, r_foot_up)
    body_transform['RightFoot']['position'] = r_ankle_pos.tolist()
    body_transform['RightFoot']['rotation'] = matrix_to_quaternion(r_foot_rotation_matrix)

    # 右つま先
    body_transform['RightToes']['position'] = r_toe_pos.tolist()
    body_transform['RightToes']['rotation'] = matrix_to_quaternion(r_foot_rotation_matrix)

    return body_transform


def create_motion_file(body_transforms, floor_offset=0.0, frame_rate=60.0, left_handed=True) -> MotionFile:
    """
    Unity用の座標変換データをMotionFileに変換する関数
    入力：右手座標系 (OpenCV)
    出力：左手座標系 (Unity)
    
    Args:
        body_transforms: Unityに変換された座標データ
        floor_offset: 床のオフセット値
        frame_rate: フレームレート
    
    Returns:
        MotionFile: 変換されたモーションデータ
    """
    
    # ボーン名のリストを定義（HumanbodyBonesに基づく）
    bone_names = [bone.name for bone in HumanbodyBones]
    
    # タイムスタンプを生成（各フレームに対して均等な間隔）
    timestamps = [i / frame_rate for i in range(len(body_transforms))]
    
    # ボーンの位置と回転を変換
    bone_positions = []
    bone_rotations = []
    
    for body_transform in body_transforms:
        frame_positions = []
        frame_rotations = []
        
        for bone_name in bone_names:
            if bone_name in body_transform:
                # ポジション変換
                position = body_transform[bone_name].get('position', [0, 0, 0])
                position_vec3 = Vector3(
                    x=float(-position[0]),
                    y=float(position[1] + floor_offset),
                    z=float(position[2])
                )
                
                # ローテーション変換
                rotation = body_transform[bone_name].get('rotation', [0, 0, 0, 1])
                rotation_quat = Quaternion(
                    x=float(rotation[0]),
                    y=float(-rotation[1]),
                    z=float(-rotation[2]),
                    w=float(rotation[3])
                )
                
                frame_positions.append(position_vec3)
                frame_rotations.append(rotation_quat)
            else:
                # データがない場合はゼロベクトル
                frame_positions.append(Vector3(0, 0, 0))
                frame_rotations.append(Quaternion(0, 0, 0, 1))
        
        bone_positions.append(frame_positions)
        bone_rotations.append(frame_rotations)
    
    # メタデータを作成
    metadata = MetaData(
        version="1.0",
        avatarName=None,
        calibrationData=None,
        boneNames=bone_names,
        length=len(body_transforms) / frame_rate,
        frameRate=frame_rate,
        recordingDate=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        left_handed=left_handed
    )
    
    return MotionFile(
        metaData=metadata,
        timestamps=timestamps,
        bonePositions=bone_positions,
        boneRotations=bone_rotations
    )


def calculate_calibration_data(poses, verbose=1) -> CalibrationData:
    # 有効なフレームを抽出
    valid_poses = [pose for pose in poses if pose is not None and not np.isnan(pose).any()]
    
    # 有効なフレームがない場合はNoneを返す
    if not valid_poses:
        return None
    
    # NumPy配列に変換
    valid_poses = np.array(valid_poses)

    # 基準の高さを計算
    left_toe_y = valid_poses[:, KEYPOINTS_DICT['left_toe'], 1]
    right_toe_y = valid_poses[:, KEYPOINTS_DICT['right_toe'], 1]
    base_y = stats.norm.fit(np.concatenate([left_toe_y, right_toe_y], axis=0))[0]

    if verbose > 1:
        print(f"calibration data calculated with valid frames: {len(valid_poses)}/{len(poses)}, base_y={base_y}")

    # 骨格計算
    m_shoulder = (valid_poses[:,KEYPOINTS_DICT['left_shoulder']] + valid_poses[:,KEYPOINTS_DICT['right_shoulder']]) / 2
    m_hip = (valid_poses[:,KEYPOINTS_DICT['left_hip']] + valid_poses[:,KEYPOINTS_DICT['right_hip']]) / 2
    body_length = stats.norm.fit(np.linalg.norm(m_shoulder - m_hip, axis=-1))[0]

    m_ear = (valid_poses[:,KEYPOINTS_DICT['left_ear']] + valid_poses[:,KEYPOINTS_DICT['right_ear']]) / 2
    neck_to_head = stats.norm.fit(np.linalg.norm(m_ear - m_shoulder, axis=-1))[0]
    
    l_upperarm_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['left_shoulder'], KEYPOINTS_DICT['left_elbow'])
    l_lowerarm_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['left_elbow'], KEYPOINTS_DICT['left_wrist'])
    r_upperarm_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['right_shoulder'], KEYPOINTS_DICT['right_elbow'])
    r_lowerarm_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['right_elbow'], KEYPOINTS_DICT['right_wrist'])
    
    l_upperleg_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['left_hip'], KEYPOINTS_DICT['left_knee'])
    l_lowerleg_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['left_knee'], KEYPOINTS_DICT['left_ankle'])
    r_upperleg_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['right_hip'], KEYPOINTS_DICT['right_knee'])
    r_lowerleg_length = calculate_bone_length(valid_poses, KEYPOINTS_DICT['right_knee'], KEYPOINTS_DICT['right_ankle'])
    
    upperarm_length = (l_upperarm_length + r_upperarm_length) / 2 
    lowerarm_length = (l_lowerarm_length + r_lowerarm_length) / 2
    upperleg_length = (l_upperleg_length + r_upperleg_length) / 2 
    lowerleg_length = (l_lowerleg_length + r_lowerleg_length) / 2

    ankle_height = 0.08 # 8cmとして計算
    pelvis_height = ankle_height + lowerleg_length + upperleg_length + body_length * 0.1 - base_y
    shoulder_height = pelvis_height + body_length * 0.9
    height = shoulder_height + neck_to_head * 1.3

    return CalibrationData(
        height=float(height),
        pelvis_height=float(pelvis_height),
        shoulder_height=float(shoulder_height),
        upperarm_length=float(upperarm_length),
        lowerarm_length=float(lowerarm_length),
        upperleg_length=float(upperleg_length),
        lowerleg_length=float(lowerleg_length)
    )

def calculate_bone_length(poses, joint_index_1, joint_index_2):
    distances = np.linalg.norm(poses[:, joint_index_1] - poses[:, joint_index_2], axis=-1)

    distances = distances[np.isfinite(distances)]  # 非有限値（NaN、inf）を除外
    
    # データが存在しない場合はゼロまたは適当なデフォルト値を返す
    if len(distances) == 0:
        return -1
    
    mu, std = stats.norm.fit(distances)
    return mu


def convert_to_motion_file(pkl_path, output_path):

    with open(pkl_path, 'rb') as p:
        data = pickle.load(p)

    # 一時データの計算
    body_transforms = []
    logs = { "None": [], "NaN": [] }

    for frame_i, pose3d in tqdm(enumerate(data['pose3d']), file=sys.stdout):

        if pose3d is None:
            body_transforms.append({})
            logs['None'].append(frame_i)
            continue

        if np.any(np.isnan(pose3d)):
            body_transforms.append({})
            logs['NaN'].append(frame_i)
            continue
        
        body_transform = pose_to_body_transforms(pose3d)
        body_transforms.append(body_transform)

    # Unity用のMotionFileに変換
    motion_file = create_motion_file(body_transforms, floor_offset=0.1, frame_rate=60.0)    

    # キャリブレーションデータを計算
    calibration_data = calculate_calibration_data(data['pose3d'])
    
    motion_file.metaData.avatarName = "DefaultAvatar"
    motion_file.metaData.calibrationData = calibration_data

    save_motion_file(motion_file, output_path)

    print(f"[Info] motion file saved to: {output_path}")