import argparse
import os

from src.visualization.output_video_2d import visualize_pose2d_with_video
from src.visualization.output_video_3d import visualize_pose3d

def main():
    parser = argparse.ArgumentParser(description="Visualize 2D or 3D pose estimation results.")
    parser.add_argument('--mode', type=str, choices=['pose2d', 'pose3d'], required=True,
                        help="Specify the visualization mode: 'pose2d' or 'pose3d'")
    parser.add_argument('--video', type=str, help="Path to input video (required for pose2d)")
    parser.add_argument('--pkl', type=str, required=True, help="Path to pose .pkl file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output video")

    args = parser.parse_args()

    if args.mode == 'pose2d':
        if not args.video:
            parser.error("--video is required for pose2d mode.")
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        visualize_pose2d_with_video(args.video, args.pkl, args.output)

    elif args.mode == 'pose3d':
        visualize_pose3d(args.pkl, args.output)

if __name__ == '__main__':
    main()
