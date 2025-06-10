import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from dust3r.model import load_model

def run_single_image_twice(model, image_path, device, image_size=512):
    img = Image.open(image_path).convert('RGB')
    colors_np = np.array(img.resize((512, 512)))  # 和 pts3d 對應的顏色圖

    transform = T.Compose([
        T.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)
    true_shape = torch.tensor([[img.height, img.width]]).to(device)

    view = {'img': img_tensor, 'true_shape': true_shape, 'instance': 0}

    with torch.no_grad():
        res1, res2 = model(view, view)

    if not isinstance(res1, dict) or 'pts3d' not in res1:
        print(f"⚠️ 影片偵測失敗: {image_path}")
        return None

    return res1['pts3d'].squeeze(0).cpu().numpy(), colors_np


def batch_inference_all_directions(model_path, input_root, output_root, image_size=224, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        device = 'cpu'
    device = torch.device(device)

    model = load_model(model_path, device=device)
    model.eval()

    directions = ['up', 'down', 'left', 'right']

    for direction in directions:
        input_dir = os.path.join(input_root, direction)
        output_dir = os.path.join(output_root, direction)
        os.makedirs(output_dir, exist_ok=True)

        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"🚀 開始處理 {direction}，共 {len(image_files)} 張圖片")

        for fname in image_files:
            fpath = os.path.join(input_dir, fname)
            scene_name = os.path.splitext(fname)[0]
            output_path = os.path.join(output_dir, f'{scene_name}.npz')

            pts3d, color_np = run_single_image_twice(model, fpath, device, image_size)
            if pts3d is not None:
                np.savez(output_path, pts3d=pts3d, colors=color_np)
                print(f"✅ 已儲存: {output_path}")
            else:
                print(f"⚠️ 無法處理: {fpath}")

    print("🎉 所有方向圖片處理完畢！")


# ✅ 執行主程式（請確認路徑正確）
batch_inference_all_directions(
    model_path='/home/jerry-tseng/robinlab/delta/VisionProject/code_for_github/xrai_dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
    input_root='/home/jerry-tseng/2025spring/ev/fp/processed_data/glove',
    output_root='/home/jerry-tseng/2025spring/ev/fp/pts3d_data/glove',
    image_size=224,
    device='cuda'
)
