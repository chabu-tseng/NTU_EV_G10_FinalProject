# PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos

<span class="author-block">
<a target="_blank" href="https://jianghanxiao.github.io/">Hanxiao Jiang</a><sup>1,2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://haoyuhsu.github.io/">Hao-Yu Hsu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://kywind.github.io/">Kaifeng Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.linkedin.com/in/hnyu/">Hsin-Ni Yu</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign</span>

### [Website](https://jianghanxiao.github.io/phystwin-web/) | [Paper](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf) | [Arxiv](https://arxiv.org/abs/2503.17973)

### Overview
This repository contains the official implementation of the **PhysTwin** framework.

![TEASER](./assets/teaser.png)


### Setup
#### 🐧Linux Setup
```bash
# Here we use cuda-12.1
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
# Create conda environment
conda create -y -n phystwin python=3.10
conda activate phystwin

# Install the packages
# If you only want to explore the interactive playground, you can skip installing Trellis, Grounding-SAM-2, RealSense, and SDXL.
bash ./env_install/env_install.sh

# Download the necessary pretrained models for data processing
bash ./env_install/download_pretrained_models.sh
```

#### 🪟Windows Setup
Thanks to @GuangyanCai contributions, now we also have a windows setup codebase in `windows_setup` branch.

#### 🐳Docker Setup
Thanks to @epiception contributions, we now have Docker support as well.
```bash
export DOCKER_USERNAME="your_alias" # default is ${whoami} (optional)
chmod +x ./docker_scripts/build.sh
./docker_scripts/build.sh

# The script accepts architecture version from https://developer.nvidia.com/cuda-gpus as an additional argument
./docker_scripts/build.sh 8.9+PTX # For NVIDIA RTX 40 series GPUs
```

### Download Our Data
Our raw arm data: https://drive.google.com/drive/u/0/folders/1R_9r_xmO2wq8ErYOlbdQ9amSR64hAY1j

XYZ of images from results of Dust3R: https://drive.google.com/file/d/1zux3-4QVLJ41UHa1KBH2P0Up_f69ywJ1

### Data Processing
#### Process data into the format for PhysTwin data processing
Please run following commands in the base of the repo.
```bash
mkdir data/different_types/bending_arm_glove/color
mkdir data/different_types/bending_arm_glove/depth

# Color
ffmpeg -i ${DATA_DIR}/EV-ArmData/video(640x640)/glove/up.mp4 -vf scale=224:224 data/different_types/bending_arm_glove/color/0.mp4
ffmpeg -i ${DATA_DIR}/EV-ArmData/video(640x640)/glove/down.mp4 -vf scale=224:224 data/different_types/bending_arm_glove/color/1.mp4
ffmpeg -i ${DATA_DIR}/EV-ArmData/video(640x640)/glove/left.mp4 -vf scale=224:224 data/different_types/bending_arm_glove/color/2.mp4
python utils/extract_frames.py data/different_types/bending_arm_glove/color/0.mp4 data/different_types/bending_arm_glove/color/0
python utils/extract_frames.py data/different_types/bending_arm_glove/color/1.mp4 data/different_types/bending_arm_glove/color/1
python utils/extract_frames.py data/different_types/bending_arm_glove/color/2.mp4 data/different_types/bending_arm_glove/color/2

# Depth
python depth_convert.py 512_data/pts3d_data/glove/up data/different_types/bending_arm_glove/depth/0
python depth_convert.py 512_data/pts3d_data/glove/down data/different_types/bending_arm_glove/depth/1
python depth_convert.py 512_data/pts3d_data/glove/left data/different_types/bending_arm_glove/depth/left
```

#### PhysTwin data processing
The original data in each case only includes `color`, `depth`, `calibrate.pkl`, `metadata.json`. All other data are processed as below to get, including the projection, tracking and shape priors.
(Note: Be aware of the conflict in the diff-gaussian-rasterization library between Gaussian Splatting and Trellis. For data processing, you don't need to install the gaussian splatting; ignore the last section in env_install.sh)
```bash
# Process the data
python script_process_data.py

# Further get the data for first-frame Gaussian
python export_gaussian_data.py

# Get human mask data for visualization and rendering evaluation
python export_video_human_mask.py
```

### Train the PhysTwin with the data
Use the processed data to train the PhysTwin. Instructions on how to get above `experiments_optimization`, `experiments` and `gaussian_output` (Can adjust the code below to only train on several cases). After this step, you get the PhysTwin that can be used in the interactive playground.
```bash
# Zero-order Optimization
python script_optimize.py

# First-order Optimization
python script_train.py

# Inference with the constructed models
python script_inference.py

# Trian the Gaussian with the first-frame data
bash gs_run.sh
```

### Evaluate the performance of the contructed PhysTwin
To evaluate the performance of the construected PhysTwin, need to render the images in the original viewpoint (similar logic to interactive playground)
```bash
# Use LBS to render the dynamic videos (The final videos in ./gaussian_output_dynamic folder)
bash gs_run_simulate.sh
python export_render_eval_data.py
# Get the quantative results
bash evaluate.sh

# Get the qualitative results
bash gs_run_simulate_white.sh
python visualize_render_results.py
```

### Play with the Interactive Playground
Use the previously constructed PhysTwin to explore the interactive playground. Users can interact with the pre-built PhysTwin using keyboard. The next section will provide a detailed guide on how to construct the PhysTwin from the original data.

![example](./assets/sloth.gif)

Run the interactive playground with our different cases (Need to wait some time for the first usage of interactive playground; Can achieve about 37 FPS using RTX 4090 on sloth case)

```bash
python interactive_playground.py \
(--inv_ctrl) \
--n_ctrl_parts [1 or 2] \
--case_name [case_name]

# Examples of usage:
python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```
or in Docker
```bash
./docker_scripts/run.sh /path/to/data \
                        /path/to/experiments \
                        /path/to/experiments_optimization \
                        /path/to/gaussian_output \
# inside container
conda activate phystwin_env
python interactive_playground.py --inv_ctrl --n_ctrl_parts 2 --case_name double_lift_cloth_3
```

Options: 
-   --inv_ctrl: inverse the control direction
-   --n_ctrol_parts: number of control panel (single: 1, double: 2) 
-   --case_name: case name of the PhysTwin case


### Material Visualization
Experimental feature to visualize the approximated material from the constructed PhysTwin.
```bash
python visualize_material.py \
--case_name [case_name]

# Examples of usage:
python visualize_material.py --case_name double_lift_cloth_1
python visualize_material.py --case_name single_push_rope
python visualize_material.py --case_name double_stretch_sloth
```



### Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{jiang2025phystwin,
    title={PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos},
    author={Jiang, Hanxiao and Hsu, Hao-Yu and Zhang, Kaifeng and Yu, Hsin-Ni and Wang, Shenlong and Li, Yunzhu},
    journal={arXiv preprint arXiv:2503.17973},
    year={2025}
}
```
