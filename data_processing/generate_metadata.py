import os, glob, json
import cv2

def generate_metadata(data_path:str, intrinsic_path:str, extrinsic_path:str):
    """
    Generate metadata for the dataset.

    Args:
        data_path (str): Path to the dataset directory.
        intrinsic_path (str): Path to the intrinsic parameters JSON file.
        extrinsic_path (str): Path to the extrinsic parameters JSON file.
    """

    # Initialize metadata dictionary
    metadata = {
        "w": None,
        "h": None,
        "k": [],
        "w2c": [],
        "fn": [],
        "cam_id": []
    }
    # read the first image to get the width and height
    img_dir = os.path.join(data_path, "ims")
    img_0_files = sorted(glob.glob(os.path.join(img_dir, "0", "*.png"))) + sorted(glob.glob(os.path.join(img_dir, "0", "*.jpg")))
    first_frame = cv2.imread(img_0_files[0])
    w, h = first_frame.shape[1], first_frame.shape[0]
    metadata["w"] = w
    metadata["h"] = h
    # camera_ids and frame numbers
    cam_ids = sorted(os.listdir(img_dir))
    frame_ids = [os.path.basename(f) for f in img_0_files]
    num_frames = len(frame_ids)
    # read intrinsic parameters
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)
    # read extrinsic parameters
    with open(extrinsic_path, "r") as f:
        extrinsics = json.load(f)
    # store results
    metadata["k"] = [[intrinsics[id] for id in cam_ids]] * num_frames
    metadata["w2c"] = [[extrinsics[id] for id in cam_ids]] * num_frames
    metadata["fn"] = [[os.path.join(id, frame_id) for id in cam_ids] for frame_id in frame_ids]
    metadata["cam_id"] = [cam_ids] * num_frames
    # save metadata to JSON file
    metadata_path = os.path.join(data_path, "train_meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    data_path = "../data/glove"
    intrinsic_path = "intrinsic.json"
    extrinsic_path = "extrinsic.json"
    metadata = generate_metadata(data_path, intrinsic_path, extrinsic_path)
    