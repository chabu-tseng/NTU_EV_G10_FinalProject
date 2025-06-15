import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--dest', type=str)
args = parser.parse_args()

if not os.path.exists(args.dest):
    os.mkdir(args.dest)

input_dir = args.source
output_dir = args.dest
    
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        
        # Remove extension and add .png
        new_filename = os.path.splitext(filename)[0] + ".png"
        new_path = os.path.join(output_dir, new_filename)
        
        img.save(new_path, "PNG")