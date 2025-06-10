import os
import re
import numpy as np
import json
from PIL import Image
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


class ArmDataset(BaseStereoViewDataset):
    def __init__(self, *, ROOT, split, resolution=224, **kwargs):
        self.ROOT = ROOT
        self.split = split
        self.resolution = resolution

        # 自動找到所有 OK_n_85f 資料夾
        self.scene_dirs = sorted([
            d for d in os.listdir(ROOT)
            if os.path.isdir(os.path.join(ROOT, d)) and re.match(r"^OK_\d+_85f$", d)
        ])
        assert self.scene_dirs, f"❌ 找不到 OK_?_85f 子資料夾 in {ROOT}"

        # 建立 (scene_name, t) 索引
        self.entries = []
        self.scene_meta = {}

        for scene in self.scene_dirs:
            scene_path = os.path.join(ROOT, scene)
            img_dir = os.path.join(scene_path, "image")
            mask_dir = os.path.join(scene_path, "mask")
            npz_dir = os.path.join(scene_path, "output_npz")
            json_path = os.path.join(scene_path, "arm_motion_pca_stable.json")

            with open(json_path, "r") as f:
                motion_data = json.load(f)

            num_frames = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
            for t in range(1, num_frames-1):
                # forward: 確保 t+1 ~ t+10 都有對應的 npz
                if t + 10 < num_frames:
                    valid = True
                    for i in range(1, 11):
                        npz_check = os.path.join(npz_dir, f"{t+i:03d}.npz")
                        if not os.path.exists(npz_check):
                            valid = False
                            break
                    if valid:
                        self.entries.append((scene, t, False))  # forward

                # reverse: 確保 t-1 ~ t-10 都有對應的 npz
                if t - 10 >= 0:
                    valid = True
                    for i in range(1, 11):
                        npz_check = os.path.join(npz_dir, f"{t-i:03d}.npz")
                        if not os.path.exists(npz_check):
                            valid = False
                            break
                    if valid:
                        self.entries.append((scene, t, True))  # reverse


            self.scene_meta[scene] = {
                "image_dir": img_dir,
                "mask_dir": mask_dir,
                "output_npz_dir": npz_dir,
                "motion_data": motion_data
            }

        super().__init__(split=split, resolution=resolution, **kwargs)
        self.scenes = [f"{s}_{t}_{'rev' if r else 'fwd'}" for s, t, r in self.entries]


    def __len__(self):
        return len(self.entries)

    def _get_views(self, scene_name, t, resolution, rng, reverse):

        meta = self.scene_meta[scene_name]

        img_path = os.path.join(meta["image_dir"], f"{t:03d}.png")
        mask_path = os.path.join(meta["mask_dir"], f"{t:03d}.png")
        npz_path = os.path.join(meta["output_npz_dir"], f"{t:03d}.npz")
        npz_gt_path = os.path.join(meta["output_npz_dir"], f"{t:03d}.npz")

        # Load data
        img_npz = np.load(npz_path)
        pts3d = img_npz["pts3d"].astype(np.float32)
        current_colors = img_npz["colors"].astype(np.float32) / 255.0
        img_pil = Image.fromarray((current_colors * 255).astype(np.uint8))
        
                # 嘗試載入連續 10 幀的顏色資料
        gt_colors_seq = []
        for i in range(1, 11):  # t+1 ~ t+10 或 t-1 ~ t-10
            target_idx = t - i if reverse else t + i
            frame_str = f"{target_idx:03d}"
            npz_path_i = os.path.join(meta["output_npz_dir"], f"{frame_str}.npz")

            # if not os.path.exists(npz_path_i):
            #     return None  # ❗ 不滿足 10 幀就丟棄這筆資料

            npz_data_i = np.load(npz_path_i)
            colors_i = npz_data_i["colors"].astype(np.float32) / 255.0
            gt_colors_seq.append(colors_i)

        # 如果真的讀到 10 幀，組成 numpy array
        gt_colors = np.stack(gt_colors_seq)  # shape = [10, H, W, 3]


        gt_npz = np.load(npz_gt_path)
        gt_pts3d = gt_npz["pts3d"].astype(np.float32)
        # gt_colors = gt_npz["colors"].astype(np.float32) / 255.0

        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask) > 127

        # Dummy intrinsics & pose
        H, W = pts3d.shape[:2]
        fx = fy = 100.0
        cx, cy = W / 2, H / 2
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)

        #print(t)
        if t >= len(meta["motion_data"]):
            raise IndexError(f"t={t} out of range for motion_data (len={len(meta['motion_data'])})")

        angle_diff = meta["motion_data"][t]["angle_diff"]
        if reverse:
            angle_diff *= -1

        angle_orig = meta["motion_data"][t]["angle_from"]
        angle_target = meta["motion_data"][t]["angle_to"]

        view = {
            "img": img_pil,
            "depthmap": pts3d[:, :, 2],
            "camera_intrinsics": intrinsics,
            "camera_pose": pose,
            "dataset": "arm_dataset",
            "label": scene_name,
            "instance": t,
            "gt_pts3d": gt_pts3d,
            "gt_colors": gt_colors,
            "angle_diff": np.float32(angle_diff),
            "angle_orig": np.float32(angle_orig),
            "angle_target": np.float32(angle_target),
            "mask": mask_np
        }

        return [view.copy(), view.copy()]   
