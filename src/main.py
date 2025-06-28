import os
import json
import time

from src.preprocessing.preprpcess import preprocess_videos
from src.pose_estimation.pose_estimator_2d import process_video
from src.pose_estimation.pose_estimator_3d import process_3d
from src.postprocessing.postprocess import refine_pose3d, filter_pose3d
from src.motion.utils import convert_to_motion_file


def main():

    # load meta data
    with open("config.json", "r") as f:
        metadata = json.load(f)

    # audio matching and trim videos
    video_processed_results = preprocess_videos(metadata=metadata)

    # initialize folders
    output_dir = metadata.get("output_dir", "output")    
    pose2d_output_dir_base = os.path.join(output_dir, "pose2d")
    pose3d_output_dir_base = os.path.join(output_dir, "pose3d")
    motion_output_dir_base = os.path.join(output_dir, "motions")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pose2d_output_dir_base, exist_ok=True)
    os.makedirs(pose3d_output_dir_base, exist_ok=True)
    os.makedirs(motion_output_dir_base, exist_ok=True)

    video_dirs = video_processed_results.get("video_dirs", [])

    model_complexity = metadata.get("model_complexity", 2)
    use_refine = metadata.get("use_refine", False)
    use_smoothing = metadata.get("use_smoothing", True)

    # process each videos
    for video_dir in video_dirs:

        base_id = os.path.basename(video_dir)

        meta = video_processed_results.get("metadata", {}).get(video_dir, None)
        if meta is None:
            print(f"[Error] meta file not found: {video_dir}")

        pose2d_output_dir = os.path.join(pose2d_output_dir_base, base_id)
        pose3d_output_dir = os.path.join(pose3d_output_dir_base, base_id)
        motion_output_dir = os.path.join(motion_output_dir_base, base_id)
        os.makedirs(pose2d_output_dir, exist_ok=True)
        os.makedirs(pose3d_output_dir, exist_ok=True)
        os.makedirs(motion_output_dir, exist_ok=True)

        
        # --- Step 1. 2d pose estimation ---
        print("[Info] Step 1: 2D Pose estimation with Mediapipe...")  

        for camera_id, video in meta.items():
            video_path = video.get("video_path", None)
            if video_path is None or not os.path.exists(video_path):
                print(f"[Error] video does not exists: {video_path}")
                break
            
            filename, ext = os.path.splitext(os.path.basename(video_path))
            pose2d_output_path = os.path.join(pose2d_output_dir, f"pose2d_{filename}.pkl")

            video.setdefault('filename', filename)
            video.setdefault('pose2d_output_path', pose2d_output_path)

            ## とりあえず、すでにファイルがある場合は実行しない
            if os.path.exists(pose2d_output_path):
                print(f"[Info] skipped 2d pose process {base_id}-{camera_id}: file already exists.")
                continue

            process_start_time = time.time()

            process_video(video_path, pose2d_output_path, model_complexity=model_complexity)
            print(f"[Info] 2D pose estimation end in {time.time() - process_start_time:.1f} sec.")

        
        # --- Step 2. 3d pose reconstruction ---
        pose3d_output_path = os.path.join(pose3d_output_dir, "pose3d.pkl")
        if not os.path.exists(pose3d_output_path):
            print("[Info] Step 2: 3D Pose estimation processing...")
            process_start_time = time.time()

            process_3d(meta.values(), pose3d_output_path)
            print(f"[Info] 3D pose estimation end in {time.time() - process_start_time:.1f} sec.")
        else:
            print("[Info] skipped 3D reconstruction. pose3d file has already exists.")


        # --- Step 3. postprocess and convert to motion file ---
        if use_refine:
            refined_output_filepath = os.path.join(pose3d_output_dir, "pose3d_refined.pkl")
            if not os.path.exists(refined_output_filepath):
                print("[Info] Step 5: refine 3d pose...")
                process_start_time = time.time()

                refine_pose3d(meta.values(), pose3d_output_path, refined_output_filepath)
                print(f"[Info] refine end in {time.time() - process_start_time:.1f} sec.")
            else:
                print("[Info] skipped refine process. refined file has already exists.")
        else:
            refined_output_filepath = pose3d_output_path
            print("[Info] skipped refine process.")


        if use_smoothing:
            filtered_output_filepath = os.path.join(pose3d_output_dir, "pose3d_filtered.pkl")
            if not os.path.exists(filtered_output_filepath):
                print("[Info] Step 6: filtering 3d pose...")
                process_start_time = time.time()

                filter_pose3d(refined_output_filepath, filtered_output_filepath)
                print(f"[Info] filtering end in {time.time() - process_start_time:.1f} sec.")
            else:
                print("[Info] skipped filtering process. filtered file has already exists.")
        else:
            filtered_output_filepath = refined_output_filepath


        output_motion_file_path = os.path.join(motion_output_dir, f"{base_id}.mot")
        if not os.path.exists(output_motion_file_path):
            convert_to_motion_file(filtered_output_filepath, output_motion_file_path)

        else:
            print("[Warning] motion file has already exists.")

        
if __name__ == '__main__':
    main()