import argparse
import open3d as o3d
import numpy as np

parser = argparse.ArgumentParser(description="Generate a point cloud from a mesh file.")
parser.add_argument("--mesh_file", type=str, default="./raw_data/bend/arm.obj", help="Path to the mesh file (obj).")
parser.add_argument("--num_points", type=int, default=2000, help="Number of points to sample from the mesh.")
parser.add_argument("--output_file", type=str, default="../data/bend/init_pt_cld.npz", help="Path to save the generated point cloud (npz).")
args = parser.parse_args()
# Load the mesh file
mesh = o3d.io.read_triangle_mesh(args.mesh_file)
pcd = mesh.sample_points_uniformly(number_of_points=args.num_points)
color = [0.831, 0.698, 0.447]
pcd.paint_uniform_color(color)
# visualize the point cloud
o3d.visualization.draw_geometries([pcd])
# Save the point cloud to a npz file
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
alpha = np.ones((points.shape[0], 1))  # Add alpha channel
points = np.concatenate((points, colors, alpha), axis=1)  # Concatenate points and colors
np.savez(args.output_file, data=points)
print(f"Point cloud saved to {args.output_file}")
