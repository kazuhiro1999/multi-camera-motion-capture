'''
Unity用モーションデータ
※サーバー側実装のバージョンと同期
'''

import json
import struct
from enum import Enum
from dataclasses import dataclass
from typing import List
import gzip

class HumanbodyBones(Enum):
    Hips = 0, 
    LeftUpperLeg = 1, 
    RightUpperLeg = 2,
    LeftLowerLeg = 3,
    RightLowerLeg = 4,
    LeftFoot = 5,
    RightFoot = 6,
    Spine = 7,
    Chest = 8,
    Neck = 9,
    Head = 10,
    LeftShoulder = 11,
    RightShoulder = 12,
    LeftUpperArm = 13,
    RightUpperArm = 14,
    LeftLowerArm = 15,
    RightLowerArm = 16,
    LeftHand = 17,
    RightHand = 18,
    LeftToes = 19,
    RightToes = 20,
    LeftEye = 21,
    RightEye = 22,
    Jaw = 23,
    LeftThumbProximal = 24,
    LeftThumbIntermediate = 25,
    LeftThumbDistal = 26,
    LeftIndexProximal = 27,
    LeftIndexIntermediate = 28,
    LeftIndexDistal = 29,
    LeftMiddleProximal = 30,
    LeftMiddleIntermediate = 31,
    LeftMiddleDistal = 32,
    LeftRingProximal = 33,
    LeftRingIntermediate = 34,
    LeftRingDistal = 35,
    LeftLittleProximal = 36,
    LeftLittleIntermediate = 37,
    LeftLittleDistal = 38,
    RightThumbProximal = 39,
    RightThumbIntermediate = 40,
    RightThumbDistal = 41,
    RightIndexProximal = 42,
    RightIndexIntermediate = 43,
    RightIndexDistal = 44,
    RightMiddleProximal = 45,
    RightMiddleIntermediate = 46,
    RightMiddleDistal = 47,
    RightRingProximal = 48,
    RightRingIntermediate = 49,
    RightRingDistal = 50,
    RightLittleProximal = 51,
    RightLittleIntermediate = 52,
    RightLittleDistal = 53,
    UpperChest = 54,
    #LastBone = 55

@dataclass
class Vector3:
    x: float
    y: float
    z: float

@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

@dataclass
class CalibrationData:
    height: float
    pelvis_height: float
    shoulder_height: float
    upperarm_length: float
    lowerarm_length: float
    upperleg_length: float
    lowerleg_length: float

@dataclass
class MetaData:
    version: str
    avatarName: str
    calibrationData: CalibrationData
    boneNames: List[str]
    length: float
    frameRate: float
    recordingDate: str
    left_handed: bool = True

@dataclass
class MotionFile:
    metaData: MetaData
    timestamps: List[float]
    bonePositions: List[List[Vector3]]
    boneRotations: List[List[Quaternion]]

def read_avatar_motion_data(file_path: str) -> MotionFile:
    with gzip.open(file_path, 'rb') as f:
        # ヘッダを検証
        identifier = f.read(4)
        if identifier != b'MOT\x00':
            raise ValueError("Invalid file identifier")

        version = struct.unpack('i', f.read(4))[0]
        if version not in [1]:  # サポートするバージョンを指定
            raise ValueError(f"Unsupported file version: {version}")
        
        # メタデータの長さを読み取る
        metadata_length = struct.unpack('i', f.read(4))[0]
        
        # メタデータを読み取り、JSONとしてパース
        metadata_json = f.read(metadata_length).decode('utf-8')
        metadata_dict = json.loads(metadata_json)
        
        # MetaDataオブジェクトを作成
        metadata = MetaData(
            version=metadata_dict['version'],
            avatarName=metadata_dict['avatarName'],
            calibrationData=CalibrationData(**metadata_dict['calibrationData']),
            boneNames=metadata_dict['boneNames'],
            length=metadata_dict['length'],
            frameRate=metadata_dict['frameRate'],
            recordingDate=metadata_dict['recordingDate'],
            left_handed=metadata_dict.get('left_handed', True)
        )
        
        bone_count = len(metadata.boneNames)
        timestamps = []
        bone_positions = []
        bone_rotations = []
        
        # タイムスタンプとボーン回転データを読み取る
        while True:
            timestamp_data = f.read(4)
            if not timestamp_data:
                break
            
            timestamp = struct.unpack('f', timestamp_data)[0]
            timestamps.append(timestamp)
            
            frame_positions = []
            for _ in range(bone_count):
                position_data = struct.unpack('fff', f.read(12))
                frame_positions.append(Vector3(*position_data))
            
            bone_positions.append(frame_positions)

            frame_rotations = []
            for _ in range(bone_count):
                rotation_data = struct.unpack('ffff', f.read(16))
                frame_rotations.append(Quaternion(*rotation_data))
            
            bone_rotations.append(frame_rotations)
        
        return MotionFile(metadata, timestamps, bone_positions, bone_rotations)
    

def save_motion_file(motion_file: MotionFile, file_path: str) -> None:
    """
    MotionFileをgzip圧縮されたバイナリファイルに保存する関数
    
    Args:
        motion_file: 保存するMotionFileオブジェクト
        file_path: 保存先のファイルパス
    """
    # キャリブレーションデータが設定されているかチェック
    if motion_file.metaData.calibrationData is None:
        raise ValueError("Calibration data is not set. Please provide valid calibration parameters before saving.")
    
    with gzip.open(file_path, 'wb') as f:
        # ヘッダを書き込む
        f.write(b'MOT\x00')  # 識別子
        f.write(struct.pack('i', 1))  # バージョン番号

        # メタデータをJSONにシリアライズ
        metadata_dict = {
            "version": motion_file.metaData.version,
            "avatarName": motion_file.metaData.avatarName,
            "calibrationData": motion_file.metaData.calibrationData.__dict__,
            "boneNames": motion_file.metaData.boneNames,
            "length": motion_file.metaData.length,
            "frameRate": motion_file.metaData.frameRate,
            "recordingDate": motion_file.metaData.recordingDate
        }
        metadata_json = json.dumps(metadata_dict).encode('utf-8')
        
        # メタデータの長さを書き込む
        f.write(struct.pack('i', len(metadata_json)))
        
        # メタデータを書き込む
        f.write(metadata_json)
        
        # 各フレームのデータを書き込む
        for timestamp, positions, rotations in zip(
            motion_file.timestamps, 
            motion_file.bonePositions, 
            motion_file.boneRotations
        ):
            # タイムスタンプを書き込む
            f.write(struct.pack('f', timestamp))
            
            # ボーンの位置を書き込む
            for position in positions:
                f.write(struct.pack('fff', position.x, position.y, position.z))
            
            # ボーンの回転を書き込む
            for rotation in rotations:
                f.write(struct.pack('ffff', rotation.x, rotation.y, rotation.z, rotation.w))


def trim_motion_file(motion_file: MotionFile, start_time: float, end_time: float) -> MotionFile:
    """
    MotionFileを指定した秒数でトリミングする関数。
    
    Parameters:
        motion_file (MotionFile): 元のMotionFileオブジェクト。
        start_time (float): トリミングの開始時間（秒）。
        end_time (float): トリミングの終了時間（秒）。
    
    Returns:
        MotionFile: トリミングされたMotionFileオブジェクト。
    """
    # フレームレートを取得
    frame_rate = motion_file.metaData.frameRate

    # 対応するフレームインデックスを計算
    start_index = max(0, int(start_time * frame_rate))
    end_index = min(len(motion_file.timestamps), int(end_time * frame_rate))

    # トリミング対象のデータをスライス
    trimmed_timestamps = motion_file.timestamps[start_index:end_index]
    trimmed_bone_positions = motion_file.bonePositions[start_index:end_index]
    trimmed_bone_rotations = motion_file.boneRotations[start_index:end_index]

    # タイムスタンプを0から始まるように調整
    start_offset = trimmed_timestamps[0]
    adjusted_timestamps = [t - start_offset for t in trimmed_timestamps]

    # トリミング後のメタデータを更新
    trimmed_metadata = MetaData(
        version=motion_file.metaData.version,
        avatarName=motion_file.metaData.avatarName,
        calibrationData=motion_file.metaData.calibrationData,
        boneNames=motion_file.metaData.boneNames,
        length=adjusted_timestamps[-1],  # 最後のタイムスタンプが長さ
        frameRate=motion_file.metaData.frameRate,
        recordingDate=motion_file.metaData.recordingDate
    )

    # 新しいMotionFileを作成して返す
    return MotionFile(
        metaData=trimmed_metadata,
        timestamps=adjusted_timestamps,
        bonePositions=trimmed_bone_positions,
        boneRotations=trimmed_bone_rotations
    )

def add_header_to_existing_file(input_file):
    """
    既存のバイナリファイルにヘッダ(identifier)を追加する。
    """
    
    with gzip.open(input_file, 'rb') as f:
        original_data = f.read()

    with gzip.open(input_file, 'wb') as f:
        f.write(b'MOT\x00')  # 識別子
        f.write(struct.pack('i', 1))  # バージョン番号 
        f.write(original_data)

    print(f"ヘッダを追加しました。")


def remove_header_to_existing_file(input_file):
    """
    既存のバイナリファイルにヘッダ(identifier)を追加する。
    """

    with gzip.open(input_file, 'rb') as f:
        identifier = f.read(4)
        if identifier != b'MOT\x00':
            raise ValueError("識別子が一致しません。ヘッダーが無いか、不正なファイルです。")        
        version = struct.unpack('i', f.read(4))[0]
        print(f"識別子: {identifier}, バージョン: {version}")
        original_data = f.read()

    with gzip.open(input_file, 'wb') as f:
        f.write(original_data)

    print(f"ヘッダを削除しました。")


if __name__ == '__main__':

    #remove_header_to_existing_file(r"C:\Users\xr\esaki\Python Projects\dance-feedback-tool-2024\motion-capture-multi-camera\output\keito_fixed.mot")
    
    # 使用例
    motion_data = read_avatar_motion_data("reference_ws2023.mot")

    print(f"Length: {motion_data.timestamps[-1]}")
    print(f"Number of frames: {len(motion_data.timestamps)}")
    print(f"Number of bones: {len(motion_data.metaData.boneNames)}")
    print(f"Frame Rate: {motion_data.metaData.frameRate}")
    print(f"First bone position: {motion_data.bonePositions[0][0]}")
    print(f"First bone rotation: {motion_data.boneRotations[0][0]}")

    # トリミング 
    #motion_trim = trim_motion_file(motion_data, 2.0, 38.0)
    #save_motion_file(motion_trim, r"C:\Users\xr\esaki\Python Projects\dance-feedback-tool-2024\motion-capture-multi-camera\output\keito_fixed_trim.mot")