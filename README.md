# 🎥 3D Human Motion Capture from Multi-View Videos

This project extracts **3D human motion** from **multi-view video footage** synchronized by audio. The final output is a `.mot` file representing motion data, generated through 2D/3D pose estimation and optional postprocessing steps.  

## ✅ Requirements

- Python 3.10 
- Dependencies (listed in `requirements.txt`):
  - ffmpeg
  - librosa
  - opencv
  - numpy
  - scipy
  - mediapipe

## 🔧 Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```  

2. Prepare your video and audio files, camera calibration settings, and create a configuration file config.json (see below).

3. Run the main script:
   ```bash
   python main.py
   ```  

## ⚙️ Configuration: config.json
   ```json
   {
     "audio": {
       "reference_audio_path": "path/to/reference audio.wav"
     },
     "videos": [
       {
         "camera_id": "camera_view_id",
         "video_path": "path/to/video.mp4",
         "camera_setting_path": "path/to/camera_setting.json"
       }, 
       ...
     ],
     "output_dir": "output",
     "max_segments_count": 1,
     "detection_overlap_mergin": 30.0,
     "video_trimming_mergin": 5.0,
     "model_complexity": 1,
     "use_refine": false,
     "use_smoothing": true
   }
   ```

### Parameter Descriptions
  - reference_audio_path: Audio file used for syncing all camera videos.
  - videos: List of video inputs with camera ID and settings.
  - camera_setting_path: Camera calibration JSON file. Must be updated when camera or resolution changes.
  - output_dir: Output root directory.
  - max_segments_count: Max detection segments count.
  - detection_overlap_mergin: Margin (in px) for overlapping detection across views.
  - video_trimming_mergin: Extra trimming time (in sec) for synced video clips.
  - model_complexity: Pose model accuracy (0: Lite, 1: Full, 2: Heavy).
  - use_refine: Whether to run 3D pose refinement (optional).
  - use_smoothing: Whether to apply smoothing filter on 3D poses.
  

## 📌 Processing Pipeline
1. Audio Sync & Trimming
  - All videos are aligned using the reference audio.

2. 2D Pose Estimation (MediaPipe)
  - 2D joint positions are estimated frame-by-frame for each camera.

3. 3D Pose Reconstruction
  - 3D joint positions are reconstructed from multiple 2D views using calibration data.

4. Optional Postprocessing
  - Pose refinement and smoothing can be applied if enabled.

5. Motion Export
  - The final result is exported as a .mot motion file.

## 📤 Output Structure
  ```bash
    output/
    ├── videos/             # synced videos 
    ├── pose2d/             # 2D pose temporaly files per camera
    ├── pose3d/             # Combined 3D pose files (raw, refined, filtered)
    ├── motions/            # Final .mot motion files
    │   └── {base_id}.mot

  ```

## 🔎 Notes
- If a .mot file already exists, it will be skipped.
- The system currently processes only the first entry in video_dirs (can be modified).
- Camera setting files (camera_setting_*.json) must be updated if the resolution or camera changes.
- For best accuracy, ensure only one subject is visible in the scene.
