import cv2
import os
import argparse
import supervision as sv
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path:str, output_path:str, crop:tuple=None, resize:tuple=None):
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Directory to save the extracted frames.
        crop (tuple): Coordinates for cropping (width, height).
        resize (tuple): New size for the frames (width, height).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Open the video
    # cap = cv2.VideoCapture(video_path)
    source_frames = Path(output_path)
    video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)
    with sv.ImageSink(target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)
    # frame_idx = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break  # End of video

    #     # Crop the frame
    #     if crop:
    #         frame_h, frame_w = frame.shape[:2]
    #         x, y = (frame_w - crop[0]) // 2, (frame_h - crop[1]) // 2
    #         w, h = crop
    #         frame = frame[y:y+h, x:x+w]

    #     # Resize the frame
    #     if resize:
    #         frame = cv2.resize(frame, resize)

    #     # Save frame as image
    #     frame_filename = os.path.join(output_path, f'{frame_idx:03d}.png')
    #     cv2.imwrite(frame_filename, frame)
    #     frame_idx += 1

    # Release the video object
    # cap.release()
    # print(f"Extracted {frame_idx} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video_path", type=str, default="raw_data/up.mp4", help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, default="data", help="Directory to save the extracted frames.")
    parser.add_argument("--crop", type=int, nargs=2, default=None, help="Crop size (width height).")
    parser.add_argument("--resize", type=int, nargs=2, default=None, help="Resize size (width height).")
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_path, crop=args.crop, resize=args.resize)
