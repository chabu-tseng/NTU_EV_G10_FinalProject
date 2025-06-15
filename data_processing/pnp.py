import cv2
import json
import numpy as np

image_path = "./raw_data/cube2/right.jpg"
# check the path
if not cv2.os.path.exists(image_path):
    raise FileNotFoundError(f"Image file {image_path} does not exist.")
mode = "3"

json_file = "./intrinsic.json"
with open(json_file, 'rb') as jf:
    intrinsic_data = json.load(jf)
intrinsic_matrix = np.array(intrinsic_data[mode], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

image = cv2.imread(image_path)
display_image = image.copy()
clicked_points_2d = []
points_3d = []

def click_event(event, x, y, flags, param):
    global clicked_points_2d, display_image, points_3d

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x={x}, y={y})")
        clicked_points_2d.append([x, y])
        display_image = image.copy()
        for pt in clicked_points_2d:
            cv2.circle(display_image, tuple(pt), 5, (0, 0, 255), -1)

        # Ask for 3D point
        try:
            coords = input("Enter 3D world coordinates for this point (format: X Y Z): ")
            x3d, y3d, z3d = map(float, coords.strip().split())
            points_3d.append([x3d, y3d, z3d])
        except Exception as e:
            print("Invalid input. Try again.")
            clicked_points_2d.pop()

def main():
    global display_image

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click_event)

    print("Click points on the image. Press 'q' to finish and solve PnP.")

    while True:
        cv2.imshow("Image", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(clicked_points_2d) < 4:
        print("Need at least 4 point correspondences for PnP.")
        return

    # Convert to numpy arrays
    object_points = np.array(points_3d, dtype=np.float64)
    image_points = np.array(clicked_points_2d, dtype=np.float64)

    # Run solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsic_matrix, dist_coeffs)

    if success:
        print("\nPnP solution found.")
        R, _ = cv2.Rodrigues(rvec)
        print("Rotation matrix:\n", R)
        print("Translation vector:\n", tvec)
    else:
        print("PnP failed.")

if __name__ == "__main__":
    main()