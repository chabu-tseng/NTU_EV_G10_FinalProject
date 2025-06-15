import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

intrinsics = []
for intrinsic in data.values():
    intrinsic[0][0] *= 512 / 640 # fx
    intrinsic[1][1] *= 512 / 640 # fy
    intrinsic[0][2] *= 512 / 640 # cx
    intrinsic[1][2] *= 512 / 640 # cy
    
    intrinsics.append(intrinsic)

print(intrinsics)