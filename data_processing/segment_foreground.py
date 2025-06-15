import os, glob, sys
import torch
code_dir = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(code_dir, "sam2")
sys.path.insert(0, sam2_path)
from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore
from PIL import Image
from tqdm import tqdm

def generate_masks(image_path:str, predictor:SAM2ImagePredictor):
    img_dirs = sorted(glob.glob(os.path.join(image_path, "ims", "*")))
    for image_dir in img_dirs:
        mask_dir = os.path.join(image_dir, os.pardir, os.pardir, "seg", os.path.basename(image_dir))
        os.makedirs(mask_dir, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        pbar = tqdm(imgs, desc=f"Processing {image_dir}", unit="image")
        for img in pbar:
            pbar.set_postfix_str(os.path.basename(img))
            image = Image.open(img).convert("RGB")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, _, _ = predictor.predict(multimask_output=True)
            # save mask as black and white image
            mask = masks[1].astype("uint8") * 255
            mask = Image.fromarray(mask)
            mask.save(os.path.join(mask_dir, os.path.basename(img)))

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(code_dir, "./sam2/checkpoints/sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    image_dir = os.path.join(code_dir, os.pardir, "data/bend")
    os.makedirs(os.path.join(image_dir, "seg"), exist_ok=True)
    generate_masks(image_dir, predictor)
