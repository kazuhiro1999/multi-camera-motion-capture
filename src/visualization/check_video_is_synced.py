"""
動画が同期されているかを確認するためのプログラム
"""
from moviepy import VideoFileClip, clips_array

# 4つの動画を読み込む（ファイルパスは適宜変更）
clip1 = VideoFileClip("synchronized_videos/temp_cFL.mp4")
clip2 = VideoFileClip("synchronized_videos/temp_cFR.mp4")
clip3 = VideoFileClip("synchronized_videos/temp_cBL.mp4")
clip4 = VideoFileClip("synchronized_videos/temp_cBR.mp4")

# すべての動画の長さを統一（最短の動画に合わせる）
min_duration = min(clip1.duration, clip2.duration, clip3.duration, clip4.duration)
clips = [clip.subclip(0, min_duration) for clip in [clip1, clip2, clip3, clip4]]

# グリッド状に並べる（2×2）
grid = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])

# 書き出し
grid.write_videofile("synchronized_videos/grid_1222.mp4", codec="libx264", fps=24)
