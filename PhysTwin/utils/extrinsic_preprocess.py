import json
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

extrinsics = []
for key in ['0', '1', '2']:
    extrinsic = np.array(data[key])
    extrinsics.append(extrinsic)

with open(args.output, 'wb') as f:
    pickle.dump(extrinsics, f)


