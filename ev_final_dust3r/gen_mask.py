import os
from rembg import remove
from PIL import Image
import io
import numpy as np
import cv2

input_folder = '/home/jerry-tseng/2025spring/ev/fp/data/OK_4_85f/image'
output_folder = '/home/jerry-tseng/2025spring/ev/fp/data/OK_4_85f/mask'
os.makedirs(output_folder, exist_ok=True)

def get_skin_mask(image_bgr): 
    img_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask = cv2.inRange(img_ycrcb, lower, upper)
    return skin_mask

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)


    with Image.open(input_path) as img:
            img = img.convert("RGBA")
            result = remove(img)

            if isinstance(result, bytes):
                result = Image.open(io.BytesIO(result)).convert("RGBA")

            rgba = np.array(result)
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            alpha = rgba[:, :, 3]

        
            bgr[alpha == 0] = 0 
            skin_mask = get_skin_mask(bgr)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

            cv2.imwrite(output_path, skin_mask)
            print(f"Saved skin-only mask: {output_path}")

