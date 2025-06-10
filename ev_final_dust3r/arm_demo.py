import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
import io
import imageio
import tempfile
from rembg import remove
import matplotlib.pyplot as plt
from dust3r.model import AsymmetricCroCo3DStereo, load_model
import torchvision.transforms as T
import torchvision.transforms.functional as F


def estimate_angle_from_pca(image):
        rgba = np.array(image)
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = rgba[:, :, 3]

        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_mask[alpha == 0] = 0

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :].astype(np.float32)

        if len(pts) < 5:
            return None

        mean, eigvecs = cv2.PCACompute(pts, mean=None)
        direction = eigvecs[0]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        return round(angle % 180, 2)

def run_inference(view1, view2, device: str = 'cuda', model_path='/home/jerry-tseng/robinlab/delta/VisionProject/code_for_github/dust3r/dust3r/checkpoints/arm2_rev/checkpoint-best.pth'):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        device = 'cpu'
    device = torch.device(device)
    model = load_model(model_path, device=device)
    model.eval()
    print("Model loaded and set to evaluation mode.")
    with torch.no_grad():
        output = model(view1, view2)
    pts3d_now, _, colors_next = output

    return pts3d_now, colors_next

def generate_video(frames, fps=10):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = temp_video.name
    temp_video.close() 

    imageio.mimsave(video_path, frames, fps=fps)

    return video_path




def pipeline(image_path, mode):
    input_img = Image.open(image_path).convert("RGBA")
    input_img = input_img.resize((224, 224))
    angle_orig = estimate_angle_from_pca(input_img)
    if angle_orig is None:
        return "Failed to estimate angle", None
    

    input_img = input_img.convert("RGB")
    transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(), # HWC Image [0, 255] -> CHW Tensor [0, 1]
        ])
    resized_img = transform(input_img).unsqueeze(0).to(device='cuda') 
    
    angle_changes = []
    if mode == "1 to -1":
        angle_changes = [1] * 10 #+ [-1] * 0
    elif mode == "-1 to 1":
        angle_changes = [-1] * 40 + [1] * 40
    total_frames =  len(angle_changes)
    print(f"Total frames: {total_frames}, angle_changes: {angle_changes}")
    # for i in range(total_frames):
    #     direction = angle_changes[i]
    video_pts3d = []
    video_colors = []
    for i in range(total_frames):
        print(f"resize range",resized_img.min(),resized_img.max())
        print(f"resized_img.shape: {resized_img.shape}")
        video_colors.append((resized_img.squeeze(0).cpu().numpy()*255).astype(np.uint8))
        direction = angle_changes[i]
        view = {
            "img": resized_img,
            "angle_diff": torch.tensor([direction], dtype=torch.float32).to(device='cuda'),
            "angle_orig":torch.tensor ([angle_changes[0]], dtype=torch.float32).to(device='cuda'),
            "angle_target": torch.tensor([angle_changes[0]], dtype=torch.float32).to(device='cuda'),
        }
        direction = angle_changes[0]
        pts3d, img = run_inference(view.copy(), view.copy(), device='cuda')
        img = img[0]
        video_pts3d.append(pts3d)
        print("img.shape",img.shape)
        resized_img = img.permute(0, 3, 1, 2).float()  # → (C, H, W)
        # resized_img = img.squeeze(0)
    # # Extract pointcloud and visualize
    # pts = traj['pts3d'][0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    # cols = traj['colors'][0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    # vis_img = visualize_pointcloud(pts, cols)
    video_colors_hwc = [video_color.transpose(1, 2, 0) for video_color in video_colors]  # ✅ numpy array

    output_color = generate_video(video_colors_hwc, fps=10)
    return f"{direction} degrees", f"{direction} degrees", output_color

demo = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),
        gr.Radio(["1 to -1", "-1 to 1"], label="Angle Change Mode")
    ],
    outputs=[
        gr.Text(label="Original Angle"),
        gr.Text(label="Angle Difference"),
        gr.Video(label="Resized_img")
    ],
    title="DUSt3R Arm Pose Manipulation Demo",
    description="Upload an image, estimate the arm angle, manipulate it, and visualize 3D results."
)

if __name__ == '__main__':
    demo.launch(share=False)
