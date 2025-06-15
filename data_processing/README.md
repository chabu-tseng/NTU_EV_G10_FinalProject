# Embodied_Vision_Final

## How to Run the Code
### Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
conda env create -f environment.yml
conda activate data_processing

git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .

cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ../..
```
### Pre-process: Align and Resize Videos
Use Adobe Premiere Pro to match the time and size of the videos.
### Extract Frames
Extract frames from video.
```bash
python3 extract_frames.py --video_path <video_path> --output_path <output_path>
```
### Generate Intrinsic Parameters from Calibration Video
Get intrinsic parameters from calibration video.
```bash
python3 camera_calibration.py <video_path> --output <output_path>
```
### Generate Extrinsic Parameters
Using PnP with a video with know object to get camera extrinsics.
```bash
python3 pnp.py
```
Check the result with `plot_camera.py`.
### Segment Foreground with SAM2
Run `segment_foreground.py`.
### Generate Training Metadata
Run `generate_metadata.py`.
