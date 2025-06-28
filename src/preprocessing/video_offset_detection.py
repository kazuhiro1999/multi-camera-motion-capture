import json
import os
import cv2
import ffmpeg
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def extract_audio(video_path, output_path=None, start=0.0, duration=60.0):
    """動画から最初の duration 秒の音声を抽出する (デフォルト: 最初の1分)"""
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '_firstmin.wav'
    
    (
        ffmpeg
        .input(video_path, ss=start, t=duration)
        .output(output_path, acodec='pcm_s16le', ac=1, ar='22050')  # モノラル, 22050Hz
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def compute_fingerprint(audio_data, sr=22050):
    """音声からフィンガープリントを作成"""
    # メル周波数ケプストラム係数（MFCC）を計算
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # クロマ特徴量を計算
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    
    # 特徴量の結合
    features = np.vstack([mfcc, chroma])
    
    return features

def compute_time_lag(ref_audio, target_audio, sr=22050):
    """2つの音声間のタイムラグを計算"""
    # 相互相関を計算
    correlation = signal.correlate(ref_audio, target_audio, mode='full')
    
    # 相互相関のピーク位置を検出
    max_idx = np.argmax(correlation)
    
    # タイムラグを計算（サンプル単位）
    lag_samples = max_idx - (len(ref_audio) - 1)
    
    # サンプル単位からミリ秒単位に変換
    lag_ms = lag_samples / sr * 1000
    
    return lag_ms, correlation

def find_sync_offset(target_video_path, reference_video_path, start=0.0, duration=60.0, audio_output_dir="temp"):
    """2つの動画間の同期オフセットを計算"""

    # Extract audio from video
    filename = os.path.splitext(os.path.basename(target_video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(target_video_path))  # avoid filename corresponding
    target_audio_filename = f"{parent_dir}_{filename}_{start:.1f}_{start+duration:.1f}.wav"
    target_audio_path = os.path.join(audio_output_dir, target_audio_filename)
    if not os.path.exists(target_audio_path):
        target_audio_path = extract_audio(target_video_path, target_audio_path, start=start, duration=duration)    
    
    filename = os.path.splitext(os.path.basename(reference_video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(reference_video_path))  # avoid filename corresponding
    reference_audio_filename = f"{parent_dir}_{filename}_{start:.1f}_{start+duration:.1f}.wav"
    reference_audio_path = os.path.join(audio_output_dir, reference_audio_filename)
    if not os.path.exists(reference_audio_path):
        reference_audio_path = extract_audio(reference_video_path, reference_audio_path, start=start, duration=duration)

    # Extract audio features
    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=22050)
    ref_features = compute_fingerprint(ref_audio, ref_sr)
    
    target_audio, target_sr = librosa.load(target_audio_path, sr=22050)
    target_features = compute_fingerprint(target_audio, target_sr)
        
    # Adjust length
    min_length = min(len(ref_audio), len(target_audio))
    ref_audio_trimmed = ref_audio[:min_length]
    target_audio_trimmed = target_audio[:min_length]
        
    # Compute time lag
    lag_ms, correlation = compute_time_lag(ref_audio_trimmed, target_audio_trimmed, sr=ref_sr)
            
    return lag_ms, correlation


def visualize_offsets(offsets, output_dir):
    """オフセットを視覚化"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(offsets)), offsets)
    plt.title('Time Offset')
    plt.xlabel('video index')
    plt.ylabel('offset (ms)')
    plt.xticks(range(len(offsets)), [f'video {i+1}' for i in range(len(offsets))])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'offset_visualization.png'))