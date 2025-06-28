import json
import os
import sys
import time
import ffmpeg

from src.preprocessing.music_segment_detection import detect_music_sections
from src.preprocessing.video_offset_detection import find_sync_offset


def trim_video_and_convert_to_60fps(input_path, output_path, start_time=0, duration=None):
    """
    映像と音声を同じ時間範囲で切り出し、映像は60fpsで出力。

    Parameters:
        input_path (str): 入力動画パス
        output_path (str): 出力動画パス
        start_time (float): 開始時間（秒）
        duration (float or None): 切り出し長さ (秒)。Noneなら最後まで。
    """
    # === 入力ストリーム作成 ===
    input_stream = ffmpeg.input(input_path, ss=start_time, t=duration)

    # === 出力設定 ===
    out = (
        ffmpeg
        .output(
            input_stream.video.filter('fps', fps=60).filter('setpts', 'PTS-STARTPTS'),
            input_stream.audio.filter('asetpts', 'PTS-STARTPTS'),
            output_path,
            vcodec='libx264',
            acodec='aac',
            audio_bitrate='192k',
            preset='fast',
            pix_fmt='yuv420p',
        )
    )

    # === 実行 ===
    out.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)


def preprocess_videos(metadata):

    # ファイル名などの初期設定
    audio_path = metadata.get("audio", {}).get("reference_audio_path", None)
    videos = metadata.get("videos", [])
    max_segments_count = metadata.get("max_segments_count", 100)
    detection_overlap_mergin = metadata.get("detection_overlap_mergin", 30)
    video_trimming_mergin = metadata.get("video_trimming_mergin", 5)

    output_dir = metadata.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    audio_output_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    video_output_dir = os.path.join(output_dir, "videos")    
    os.makedirs(video_output_dir, exist_ok=True)

    print(f"[Info] config loaded.")


    # --- 1. Detect music sections from 1 video ---
    reference_video = videos[0]
    reference_video_path = reference_video.get("video_path", None)
    matching_results = detect_music_sections(audio_path, reference_video_path, output_dir=output_dir)

    if len(matching_results) == 0:
        sys.exit(0)


    # --- 2. Calculate syncronization offset of other videos ---
    print(f"[Info] calculating sync offset...")

    match = matching_results[0]  # select first index (most reliable match)
    match_start = match["start_time"] - detection_overlap_mergin
    match_duration = match["duration"] + 2 * detection_overlap_mergin  # as a overlap margin

    for video in videos:
        target_video_path = video.get("video_path", None)

        if target_video_path is None or not os.path.exists(target_video_path):
            print(f"[Error] video does not exists: {target_video_path}")
            return    

        if video == reference_video:   
            video.setdefault('sync_offset', 0.0)
        else:
            lag_ms, correlation = find_sync_offset(target_video_path, reference_video_path, start=match_start, duration=match_duration, audio_output_dir=audio_output_dir)
            video.setdefault('sync_offset', float(lag_ms/1000))
        
        print(f"[Info] sync offset: {video.get('camera_id')} => {video.get('sync_offset')}")
    

    # --- 3. Trim videos using detection and sync result
    print(f"[Info] start trimming videos.")
    
    matching_results = matching_results[:max(1, min(len(matching_results), max_segments_count))]  # limit to max count 
    matching_results.sort(key=lambda x: x.get("start_time", 1e+9))  # sort by start_time
    video_processed_results = {
        "video_dirs": [],
        "metadata":{}
    }

    for i, match in enumerate(matching_results):  # for all segments
        start = match.get("start_time") - video_trimming_mergin
        duration = match.get("duration") + 2 * video_trimming_mergin  # as a trimming margin
        similarity_score = match.get("similarity_score")

        if similarity_score > 0.1:
            print(f"[Info] skipped trimming segment_{i}: poor similarity score.")
            continue

        video_dir = os.path.join(video_output_dir, f"segment_{i}")    
        os.makedirs(video_dir, exist_ok=True)

        meta = {}

        for video in videos:
            camera_id = video.get("camera_id", "unknown")
            video_path = video.get("video_path", None)
            sync_offset = video.get("sync_offset", 0.0)
            output_video_path = os.path.join(video_dir, f"{camera_id}.mp4")

            # add video meta data
            meta.setdefault(camera_id, {
                "original_video_path": video_path,
                "video_path": output_video_path,
                "sync_offset": sync_offset,
                "start_time": max(start - sync_offset, 0),
                "duration": duration,
                "camera_setting_path": video.get("camera_setting_path", None)  # copy camera setting
            })

            if os.path.exists(output_video_path):
                print(f"[Info] skipped video segment_{i} {camera_id}: already exists.")
                continue

            print(f"[Info] trimming video... segment_{i} {camera_id}")
            process_start_time = time.time()

            trim_video_and_convert_to_60fps(
                input_path=video_path,
                output_path=output_video_path,
                start_time=max(start - sync_offset, 0),
                duration=duration
            )

            print(f"[Info] trimming end in {time.time() - process_start_time:.1f} sec.")

        video_processed_results["video_dirs"].append(video_dir)
        video_processed_results["metadata"].setdefault(video_dir, meta)

        # save video meta data (.json)
        with open(os.path.join(video_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=4)

    # copy meta data to output directory
    with open(os.path.join(output_dir, "preprocess.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("[Info] preprocess completed!")

    return video_processed_results


if __name__ == '__main__':
    
    # load meta data
    with open("preprocess.json", "r") as f:
        metadata = json.load(f)

    preprocess_videos(metadata=metadata)
