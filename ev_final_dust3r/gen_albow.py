import os
import cv2
import json
from mmpose.apis import init_pose_model, inference_topdown_pose_model
from mmdet.apis import init_detector, inference_detector

input_folder = '/home/jerry-tseng/2025spring/ev/fp/data/OK_1_85f/image'
output_json = '/home/jerry-tseng/2025spring/ev/fp/data/OK_1_85f/arm_motion_right_elbow.json'

pose_config = 'https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py?raw=true'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/body_2d_keypoint/rtmpose/coco/rtmpose-m_simcc-body7_pt-aic-coco_420e-256x192-e16c7aa5_20230314.pth'

det_config = 'https://github.com/open-mmlab/mmdetection/blob/main/configs/yolox/yolox_s_8x8_300e_coco.py?raw=true'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211126_140254-1ef88d67.pth'


det_model = init_detector(det_config, det_checkpoint, device='cuda')
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda')

def get_right_elbow_mmpose(image_path):
    image = cv2.imread(image_path)
    detect_result = inference_detector(det_model, image)
    person_bboxes = [bbox for bbox in detect_result.pred_instances.cpu().numpy() if bbox[-1] == 0]

    if not person_bboxes:
        return None

    pose_results = inference_topdown_pose_model(pose_model, image, [{'bbox': person_bboxes[0][:4]}], format='xyxy')[0]
    keypoints = pose_results.pred_instances.keypoints[0].cpu().numpy()

    right_elbow = keypoints[8] 
    return (round(right_elbow[0], 2), round(right_elbow[1], 2))


files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg')))
positions = {}

for file in files:
    path = os.path.join(input_folder, file)
    elbow = get_right_elbow_mmpose(path)
    if elbow:
        positions[file] = elbow
    else:
        print(f"{file} - Detection fail")


motion_log = []
for i in range(1, len(files)):
    prev = files[i - 1]
    curr = files[i]
    if prev in positions and curr in positions:
        x0, y0 = positions[prev]
        x1, y1 = positions[curr]
        dx = round(x1 - x0, 2)
        dy = round(y1 - y0, 2)
        dist = round((dx**2 + dy**2)**0.5, 2)
        motion_log.append({
            "from": prev,
            "to": curr,
            "right_elbow_from": [x0, y0],
            "right_elbow_to": [x1, y1],
            "motion_vector": [dx, dy],
            "distance": dist
        })

# === 儲存 JSON ===
with open(output_json, 'w') as f:
    json.dump(motion_log, f, indent=2)

print(f"Save to: {output_json}")
