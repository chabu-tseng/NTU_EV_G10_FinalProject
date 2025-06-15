import os
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

"""
Hyperparam for Ground and Tracking
"""

# Put below base path into args
parser = ArgumentParser()
parser.add_argument("--video_path", type=str)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

VIDEO_PATH = args.video_path
OUTPUT_PATH = args.output_path

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

existDir(OUTPUT_PATH)


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(OUTPUT_PATH)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.png"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)
