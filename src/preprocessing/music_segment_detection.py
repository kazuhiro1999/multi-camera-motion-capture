import json
import os
import time
import numpy as np
import librosa
import ffmpeg
import matplotlib.pyplot as plt

def extract_audio_from_video(video_path):
    """動画から音声を抽出する"""
    
    # 一時ファイルに保存
    temp_audio_path = "temp/temp_audio.wav"
    os.makedirs("temp", exist_ok=True)
    
    try:
        # ffmpeg-pythonを使用して音声を抽出
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio_path, 
                    acodec='pcm_s16le',  # WAV形式で出力
                    ar=44100,            # サンプリングレート
                    ac=2,                # ステレオ
                    map='a')             # 音声ストリームのみ
            .global_args('-y')           # 既存ファイルを上書き
            .run()
        )
        
        # 音声データを読み込み
        y, sr = librosa.load(temp_audio_path, sr=None)
        
        # 一時ファイルを削除（オプション）
        os.remove(temp_audio_path)
        
        return y, sr
    except ffmpeg.Error as e:
        print(f"FFmpeg-pythonでの音声抽出エラー: {e.stderr.decode() if e.stderr else 'なし'}")
        raise

def extract_audio_features(y, sr, frame_length=2048, hop_length=512):
    """音声からMFCCなどの特徴量を抽出する"""
    # MFCC (メル周波数ケプストラム係数)を抽出
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, 
                               n_fft=frame_length, hop_length=hop_length)
    
    # スペクトル重心を抽出
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                         n_fft=frame_length, hop_length=hop_length)
    
    # 特徴量を結合
    features = np.vstack([mfcc, spectral_centroid])
    
    return features.T  # 転置して時間方向を行にする

def get_audio_energy(y, sr, frame_length=2048, hop_length=512):
    """音声のエネルギー(音量)を計算する"""
    # 短時間フーリエ変換
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    
    # エネルギーを計算
    energy = np.sum(S**2, axis=0)
    
    return energy

def find_high_energy_segments_window(energy, target_length, threshold_percentile=80, overlap_ratio=0.8):
    """
    音楽データと同じ長さのウィンドウを使って、平均音量が大きい区間を候補として抽出する
    
    Parameters:
    energy: 音量のnumpy配列
    target_length: 参照音楽データの長さ（フレーム数）
    threshold_percentile: 音量の閾値とするパーセンタイル
    overlap_ratio: 連続するウィンドウの重複比率 (0-1)
    
    Returns:
    segments: 候補区間のリスト
    """
    # ウィンドウのステップサイズを計算（重複を考慮）
    step_size = max(1, int(target_length * (1 - overlap_ratio)))
    
    # 各ウィンドウの平均音量を計算
    window_energies = []
    window_positions = []
    
    for start_pos in range(0, len(energy) - target_length + 1, step_size):
        end_pos = start_pos + target_length
        window_avg_energy = np.mean(energy[start_pos:end_pos])
        window_energies.append(window_avg_energy)
        window_positions.append(start_pos)
    
    # 平均音量の閾値を計算
    threshold = np.percentile(window_energies, threshold_percentile)
    
    # 閾値以上のウィンドウを検出
    segments = []
    for i, avg_energy in enumerate(window_energies):
        if avg_energy >= threshold:
            start_pos = window_positions[i]
            segment = list(range(start_pos, start_pos + target_length))
            segments.append(segment)

    # 類似または重複する区間をマージするオプション
    merged_segments = merge_overlapping_segments(segments)
    
    return merged_segments

def merge_overlapping_segments(segments, overlap_threshold=0.2):
    """
    重複する区間をマージする
    
    Parameters:
    segments: 区間のリスト
    overlap_threshold: マージする閾値（重複率）
    
    Returns:
    merged_segments: マージされた区間のリスト
    """
    if not segments:
        return []
    
    # 開始位置でソート
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged_segments = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        previous = merged_segments[-1]
        
        # 重複度を計算
        previous_set = set(previous)
        current_set = set(current)
        overlap_size = len(previous_set.intersection(current_set))
        overlap_ratio = overlap_size / min(len(previous), len(current))
        
        if overlap_ratio > overlap_threshold:
            # 区間をマージ
            merged = sorted(list(previous_set.union(current_set)))
            merged_segments[-1] = merged
        else:
            # 新しい区間として追加
            merged_segments.append(current)
    
    return merged_segments

def match_audio_segments(reference_features, video_features, reference_length, hop_length=512, sr=44100):
    """マッチングを行い、0.1秒以内の精度で一致する区間を検出"""
    best_matches = []
    process_start_time = time.time()
    
    # 0.1秒あたりのフレーム数
    frames_per_01sec = int(0.1 * sr / hop_length)
    
    # 動画の音量エネルギーを計算
    video_energy = np.sum(video_features**2, axis=1)
    
    # ウィンドウベースで候補区間を検出
    candidate_segments = find_high_energy_segments_window(
        video_energy, 
        reference_length,
        threshold_percentile=80,
        overlap_ratio=0.9
    )
    
    print(f"[Info] detected segments count: {len(candidate_segments)}")
    
    # 参照特徴量の準備
    ref_features_flat = reference_features.flatten()
    ref_norm = np.sqrt(np.sum(ref_features_flat**2))
    
    # 各候補区間を効率的に処理
    for i, segment in enumerate(candidate_segments):
        print(f"[Info] processing segment {i+1}/{len(candidate_segments)}")
        
        # セグメントの開始位置と終了位置
        start_idx = segment[0]
        end_idx = segment[-1]
        
        # 精度を保つため、周辺領域も含めて探索
        padding = frames_per_01sec * 10
        search_start = max(0, start_idx - padding)
        search_end = min(len(video_features), end_idx + padding)
        
        # 相関係数を使用して最も類似する区間を見つける
        best_corr = -1
        best_offset = 0
        
        # スライディングウィンドウで相関係数を計算
        for offset in range(search_start, search_end - reference_length + 1):
            # 対象区間の特徴量
            segment_features = video_features[offset:offset+reference_length].flatten()
            
            # 特徴量が十分な長さを持つか確認
            if len(segment_features) < len(ref_features_flat):
                continue
                
            # 正規化相互相関係数を計算（高速）
            segment_features = segment_features[:len(ref_features_flat)]
            segment_norm = np.sqrt(np.sum(segment_features**2))
            
            # ゼロ除算を避ける
            if segment_norm == 0:
                continue
                
            corr = np.sum(ref_features_flat * segment_features) / (ref_norm * segment_norm)
            
            # 最大相関を追跡
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        # 高相関の一致区間のみを採用
        if best_corr > 0.75:  # 閾値は調整可能
            # 開始時間と終了時間を秒単位で計算
            start_time = best_offset * hop_length / sr
            end_time = (best_offset + reference_length) * hop_length / sr
            
            # 時間ずれが0.1秒以内であることを確認
            # (正規化相関係数なので時間ずれは既に最小化されている)
            
            # 類似度スコアを計算（相関係数をそのまま使用）
            similarity_score = 1 - best_corr  # 小さいほど良い
            
            best_matches.append((start_time, end_time, similarity_score))
    
    print(f"[Info] audio matching competed in {time.time() - process_start_time:.1f} sec.")

    # 検出結果をスコアでソート
    best_matches.sort(key=lambda x: x[2])
    
    return best_matches

def detect_music_sections(reference_audio_path, video_path, output_dir="audio_matching_result", save_fig=True):
    """音楽と一致する区間を検出する"""
    # 参照音声を読み込む
    reference_y, reference_sr = librosa.load(reference_audio_path, sr=None)
    reference_length = len(reference_y)
    
    # 動画から音声を抽出
    video_y, video_sr = extract_audio_from_video(video_path)
    
    # サンプリングレートが異なる場合はリサンプリング
    if reference_sr != video_sr:
        reference_y = librosa.resample(reference_y, orig_sr=reference_sr, target_sr=video_sr)
        reference_sr = video_sr
    
    # 特徴量抽出のパラメータ
    frame_length = 2048
    hop_length = 512
    
    # 特徴量を抽出
    reference_features = extract_audio_features(reference_y, reference_sr, frame_length, hop_length)
    video_features = extract_audio_features(video_y, video_sr, frame_length, hop_length)
    
    # 参照音声の長さ（フレーム数）
    reference_frame_length = len(reference_features)
    
    # マッチングを実行
    matches = match_audio_segments(reference_features, video_features, reference_frame_length, hop_length, video_sr)
    
    if matches:
        # メモ：類似度の基準は0.1以下程度        
        # マッチング結果をJSONファイルに出力
        results = []
        for i, (start_time, end_time, score) in enumerate(matches):
            results.append({
                "match_id": i+1,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(end_time - start_time, 2),
                "similarity_score": round(score, 4)
            })

        # JSONファイルに保存
        output_json_path = os.path.join(output_dir, f"audio_matching_results.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "reference_audio_path": reference_audio_path,
                    "video_path": video_path,
                    "results": results
                }, 
                f, 
                ensure_ascii=False, indent=4)
            
        if save_fig:
            output_img_path = os.path.join(output_dir, f"audio_matching_result.png")

            # 波形表示（可視化）
            plt.figure(figsize=(10, 4))
                    
            # 音声の波形
            librosa.display.waveshow(video_y, sr=video_sr)
            plt.title('Wave and Detected Segments')
            plt.xlabel('t(sec)')
                    
            # 検出区間を表示（透明度を変えて複数区間を表示）
            for i, (start, end, score) in enumerate(matches):
                alpha = 0.5 - (i * 0.1)  # 上位の結果ほど濃く表示
                if alpha < 0.1:
                    alpha = 0.1
                plt.axvspan(start, end, color='red', alpha=alpha)
                plt.text(start, 0, f"{score:.2f}", verticalalignment='center')
                    
            plt.tight_layout()
            plt.savefig(output_img_path)
        
        return results
    else:
        print("[Error] No matched segment.")
        return []
    

if __name__ == "__main__":
    # 参照音声ファイルと動画ファイルのパスを指定
    reference_audio_path = 'audios/NEFFEX - Losing My Mind.wav'
    video_path = r"C:\Users\xr\esaki\WS2024\Videos\202412\FL\concat.mp4"
    
    # マッチングを実行
    detect_music_sections(reference_audio_path, video_path)