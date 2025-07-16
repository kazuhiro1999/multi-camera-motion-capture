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
   python run.py
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
  
## 📷 Camera Setting Format (camera_setting_*.json)
Each camera requires a calibration JSON file containing its intrinsic and extrinsic parameters. These are critical for 3D pose reconstruction.
   ```json
  {  
    "name": "DJI Osmo Action 3",
    "intrinsic_matrix": [
      [752.0, 0.0, 960.0],
      [0.0, 752.0, 540.0],
      [0.0, 0.0, 1.0]
    ],
    "extrinsic_matrix": [
      [-0.7071067861992015, -1.1314261139488288e-16, -0.7071067761738934, 1.7744794845495025e-08],
      [1.1314261139488288e-16, -1.0, 4.686520365223107e-17, 1.0000000000000002],
      [-0.7071067761738934, -4.686520365223107e-17, 0.7071067861992015, 2.5031579784263953]
    ],
    "image_width": 1920,
    "image_height": 1080
  }
   ```

### Required Fields
  - intrinsic_matrix (3x3): Camera intrinsics, typically obtained from calibration tools.
  - extrinsic_matrix (3x4): World-to-camera transformation matrix.
  - image_width, image_height: (Optional) Resolution of the input video.

💡 Note: Only intrinsic_matrix and extrinsic_matrix are strictly required.  

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
