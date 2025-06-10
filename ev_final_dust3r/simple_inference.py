import argparse
import torch
from PIL import Image
import torchvision.transforms as T
import sys
import os

torch.serialization.add_safe_globals([argparse.Namespace])
# 確保你的 DUST3r 程式碼路徑在系統路徑中，這樣才能正確導入模組
# 請將下面的 '/path/to/your/dust3r/code_for_github' 替換為你的 DUST3r 程式碼的根目錄的實際路徑
# 例如：如果你的 dust3r 程式碼放在 /home/user/my_dust3r_code，就改成 sys.path.insert(0, '/home/user/my_dust3r_code')
sys.path.insert(0, '/home/jerry-tseng/robinlab/delta/VisionProject/code_for_github/xrai_dust3r')

# 從 dust3r 模組中導入必要的類別和函數
# 這裡假設你的模型結構和 load_model 函數定義在 dust3r.model 中
try:
    from dust3r.model import AsymmetricCroCo3DStereo, load_model
    # 如果你的數據處理需要其他工具，也需要在這裡導入
    # from dust3r.utils.image import load_images # 例如，如果想用官方的 load_images
    # from dust3r.utils.device import to_numpy # 例如，如果想把結果轉為 numpy
except ImportError as e:
    print(f"Error importing DUST3r modules: {e}")
    print("Please ensure you have added the root of your DUST3r code to sys.path correctly.")
    sys.exit(1)


def run_simple_inference(model_path: str, image1_path: str, image2_path: str, image_size: int = 224, device: str = 'cuda', output_npz_path: str = "/home/jerry-tseng/robinlab/delta/VisionProject/data/scene1/predict/frame1_predict.npz"):
    """
    載入 DUST3r 模型，對兩張輸入影像執行前向計算，並印出主要輸出的張量形狀。

    Args:
        model_path: 訓練好的 DUST3r 模型 .pth 檔案路徑。
        image1_path: 第一張輸入影像的路徑。
        image2_path: 第二張輸入影像的路徑。
        image_size: 模型期望的輸入影像尺寸 (例如 224 或 512)。需與訓練時一致。
        device: 執行預測的裝置 ('cuda' 或 'cpu')。
    """
    # 檢查裝置是否可用
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        device = 'cpu'
    device = torch.device(device)

    # 1. 載入模型
    print(f"Loading model from {model_path} on device {device}...")
    try:
        model = load_model(model_path, device=device)
        model.eval() # 設定模型為評估模式
        print("Model loaded and set to evaluation mode.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 2. 準備輸入數據
    print(f"Loading and preparing images: {image1_path}, {image2_path}...")
    try:
        # 載入影像 (使用 PIL)
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')

        # 定義預處理轉換 (盡量與訓練時保持一致)
        # 這裡使用一個簡單的 resize 和 ToTensor 轉換作為範例
        # 如果你的訓練有更複雜的預處理 (例如特定的歸一化)，請在這裡添加
        transform = T.Compose([
            T.Resize((image_size, image_size)), # 調整到模型期望的尺寸
            T.ToTensor(), # HWC Image [0, 255] -> CHW Tensor [0, 1]
            # 如果訓練時有特定的歸一化，也要加在這裡
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 範例 ImageNet 歸一化
        ])

        img_tensor1 = transform(img1).unsqueeze(0).to(device) # 添加 batch 維度 (Batch, Channel, Height, Width) 並移動到裝置
        img_tensor2 = transform(img2).unsqueeze(0).to(device)

        # 構建模型 forward 函數期望的字典格式
        # 'true_shape' 通常包含原始影像的高度和寬度，模型可能在內部使用它
        view1 = {'img': img_tensor1, 'true_shape': torch.tensor([[img1.height, img1.width]]).to(device)}
        view2 = {'img': img_tensor2, 'true_shape': torch.tensor([[img2.height, img2.width]]).to(device)}

        print("Images prepared.")

    except FileNotFoundError as e:
        print(f"Error: Image file not found: {e}")
        return None
    except Exception as e:
        print(f"Error preparing images: {e}")
        return None

    # 3. 執行前向傳播 (在 torch.no_grad() 模式下禁用梯度計算)
    print("Running model inference...")
    with torch.no_grad():
        # 呼叫模型的前向方法
        # 參考 AsymmetricCroCo3DStereo 的 forward 函數簽名
        output = model(view1, view2)
    
    res1, res2, res_traj = output

    print(res_traj.keys())
    print("Inference complete. Model output structure:")

    # 4. 處理和顯示模型輸出 (簡單範例：印出主要張量的形狀)
    # DUST3r 的輸出通常是一個字典，包含不同視角和不同類型的結果
    # 檢查並印出你可能感興趣的關鍵輸出，例如 'pts3d' 和 'pts3d_seq'
    output_traj = res_traj
    main_outputs = {}
    
    if 'traj3d' in output_traj:
        result = output_traj['traj3d']
        print(f"  'traj3d' (3D points) shape: {result.shape}, dtype: {result.dtype}")

        if output_npz_path:
            try:
                # 移除批次維度 (batch dimension)，從 [1, 11, 224, 224, 3] 變成 [11, 224, 224, 3]
                pts3d_np = result.squeeze(0)
                # 將張量從 GPU 移到 CPU，並轉換為 NumPy 陣列
                pts3d_np = pts3d_np.cpu().numpy()

                # 確保目標目錄存在
                output_dir = os.path.dirname(output_npz_path)
                if output_dir: # 如果路徑中包含目錄
                     os.makedirs(output_dir, exist_ok=True)

                # 使用 numpy.savez 保存為 .npz 檔案
                # 你需要給保存的陣列一個名字，這裡使用 'pts3d'
                import numpy as np
                np.savez(output_npz_path, pts3d=pts3d_np)

                print(f"    'pts3d' saved to {output_npz_path} with shape {pts3d_np.shape}")

            except Exception as e:
                print(f"    Error saving pts3d to NPZ: {e}")
        #          if output_npz_path:
        #              try:
        #                  # 移除批次維度 (batch dimension)，從 [1, 11, 224, 224, 3] 變成 [11, 224, 224, 3]
        #                  pts3d_seq_np = pts3d_seq_v1.squeeze(0)
        #                  # 將張量從 GPU 移到 CPU，並轉換為 NumPy 陣列
        #                  pts3d_seq_np = pts3d_seq_np.cpu().numpy()

        #                  # 確保目標目錄存在
        #                  output_dir = os.path.dirname(output_npz_path)
        #                  if output_dir: # 如果路徑中包含目錄
        #                       os.makedirs(output_dir, exist_ok=True)

        #                  # 使用 numpy.savez 保存為 .npz 檔案
        #                  # 你需要給保存的陣列一個名字，這裡使用 'pts3d_sequence'
        #                  import numpy as np
        #                  np.savez(output_npz_path, pts3d_sequence=pts3d_seq_np)

        #                  print(f"    'pts3d_seq' saved to {output_npz_path} with shape {pts3d_seq_np.shape}")

        #              except Exception as e:
        #                  print(f"    Error saving pts3d_seq to NPZ: {e}")
        #          # === 保存程式碼結束 ===


        #     if 'depth' in view1_output:
        #          # ... (保存 depth 的程式碼類似) ...
        #          pass

        #     if 'conf' in view1_output:
        #          # ... (保存 conf 的程式碼類似) ...
        #          pass


    if 'view2_dust3r' in output:
        # ... (處理 View 2 的輸出，類似 View 1) ...
        pass

    return main_outputs


if __name__ == "__main__":
    # 設定命令列參數解析器
    parser = argparse.ArgumentParser(description='Run DUST3r inference on two images to get raw model output.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained DUST3r model .pth file.')
    parser.add_argument('--image1_path', type=str, required=True,
                        help='Path to the first input image.')
    parser.add_argument('--image2_path', type=str, required=True,
                        help='Path to the second input image.')
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 512],
                        help='Image size for model input (e.g., 224 or 512). Must match training.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for inference (cuda or cpu).')
    parser.add_argument('--save_pts3d_seq_npz', type=str, default=None,
                        help='Path to save the pts3d_seq output as an NPZ file.')
    # 你可以在這裡添加更多的參數，例如用於指定輸出檔案的路徑等

    args = parser.parse_args()

    # 執行簡單的推理腳本
    inference_results = run_simple_inference(
        model_path=args.model_path,
        image1_path=args.image1_path,
        image2_path=args.image2_path,
        image_size=args.image_size,
        device=args.device,
        output_npz_path=args.save_pts3d_seq_npz  # ✅ 把這行加上去！
    )

    # 如果需要進一步處理結果，可以在這裡進行
    if inference_results is not None:
        print("\nSimple inference script finished.")
        # 如果你保存了檔案，可以在這裡提醒使用者檢查
        if args.save_pts3d_seq_npz:
             print(f"Check {args.save_pts3d_seq_npz} for the saved pts3d_seq.")