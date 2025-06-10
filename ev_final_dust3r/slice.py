import os
import cv2

input_root = "/home/jerry-tseng/2025spring/ev/fp/newdata"
output_root = "/home/jerry-tseng/2025spring/ev/fp/processed_data"
subfolders = ["glove", "normal"]
video_names = ["down.mp4", "left.mp4", "right.mp4", "up.mp4"]
target_size = (512, 512)

for sub in subfolders:
    for video_name in video_names:
        video_path = os.path.join(input_root, sub, video_name)
        output_dir = os.path.join(output_root, sub, os.path.splitext(video_name)[0])

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        if not cap.isOpened():
            print(f"fail: {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, target_size)
            frame_filename = f"{frame_count:05d}.png"
            cv2.imwrite(os.path.join(output_dir, frame_filename), resized)
            frame_count += 1

        cap.release()
        print(f"Done: {video_path} -> {frame_count} frames")

print("Finish")
