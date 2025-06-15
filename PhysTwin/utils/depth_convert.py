import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--dest', type=str)
args = parser.parse_args()

if not os.path.exists(args.dest):
    os.mkdir(args.dest)
    
filenames = os.listdir(args.source)
for filename in filenames:
    filepath = os.path.join(args.source, filename)
    array = np.load(filepath)
    array = array['pts3d'][:,:,2]  # Extract the depth values

    name, ext = os.path.splitext(filename)
    new_name = str(int(name)) + '.npy'
    
    np.save(os.path.join(args.dest, new_name), array)
