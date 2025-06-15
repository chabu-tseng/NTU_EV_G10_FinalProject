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
    name, ext = os.path.splitext(filename)
    new_name = str(int(name)) + ext

    old_path = os.path.join(args.source, filename)
    new_path = os.path.join(args.dest, new_name)
    os.rename(old_path, new_path)
    
    
