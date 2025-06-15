import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load extrinsics
with open("extrinsic2.json", "r") as f:
    data = json.load(f)

# Assume same intrinsics for simplicity
fx, fy = 800, 800
cx, cy = 320, 240
image_w, image_h = 640, 480
depth = 50  # Virtual image plane depth for frustum

def get_frustum_corners(fx, fy, cx, cy, w, h, depth):
    """Generate 3D coordinates of image plane corners in camera space"""
    corners_px = np.array([
        [0, 0],         # top-left
        [w, 0],         # top-right
        [w, h],         # bottom-right
        [0, h]          # bottom-left
    ])

    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        corners_cam.append([x, y, z])
    return np.array(corners_cam)  # shape (4, 3)

def draw_frustum(ax, R, t, label=None):
    """Draw camera frustum in world space"""
    R = np.array(R)
    t = np.array(t).reshape(3)

    # Camera origin
    cam_center = -R.T @ t

    # Frustum corners in camera frame
    corners_cam = get_frustum_corners(fx, fy, cx, cy, image_w, image_h, depth)

    # Transform to world frame
    corners_world = (R.T @ (corners_cam.T - t[:, np.newaxis])).T

    # Define lines from center to corners
    for corner in corners_world:
        ax.plot(*zip(cam_center, corner), color='gray', linewidth=1)

    # Draw image plane as polygon
    verts = [corners_world]
    ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.3))

    ax.scatter(*cam_center, c='red')
    if label:
        ax.text(*cam_center, f"Cam {label}", fontsize=9)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw all camera frustums
for cam_id, T in data.items():
    # print(T)
    T = np.array(T)
    R = T[:3, :3]  # Rotation matrix
    t = T[:3, 3]  # Translation vector
    draw_frustum(ax, R, t, label=cam_id)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Frustums in World Space")
ax.view_init(elev=20, azim=45)
ax.set_box_aspect([1,1,1])
plt.tight_layout()
plt.show()