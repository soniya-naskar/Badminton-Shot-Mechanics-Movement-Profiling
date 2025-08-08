# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import math
# import time

# # ======================
# # Utility Functions
# # ======================

# def get_angle(p1, p2, p3):
#     """Calculate angle between three points."""
#     a = np.array(p1)
#     b = np.array(p2)
#     c = np.array(p3)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def annotate_pose_info(frame, keypoints, actions):
#     """Draw pose annotations and custom insights."""
#     for idx, kp in enumerate(keypoints):
#         if kp[2] > 0.3:  # confidence threshold
#             cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

#     y0 = 30
#     for i, action in enumerate(actions):
#         cv2.putText(frame, action, (10, y0 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     return frame

# def detect_actions(keypoints):
#     """Extract key biomechanical patterns based on pose keypoints."""
#     results = []

#     try:
#         left_hip = keypoints[23][:2]
#         right_hip = keypoints[24][:2]
#         left_knee = keypoints[25][:2]
#         right_knee = keypoints[26][:2]
#         left_ankle = keypoints[27][:2]
#         right_ankle = keypoints[28][:2]
#         left_shoulder = keypoints[11][:2]
#         right_shoulder = keypoints[12][:2]
#         left_elbow = keypoints[13][:2]
#         right_elbow = keypoints[14][:2]
#         left_wrist = keypoints[15][:2]
#         right_wrist = keypoints[16][:2]

#         # Posture detection
#         knee_angle = get_angle(left_hip, left_knee, left_ankle)
#         if knee_angle < 100:
#             results.append("Lunge posture detected")

#         # Arm mechanics
#         elbow_angle = get_angle(left_shoulder, left_elbow, left_wrist)
#         if elbow_angle < 100:
#             results.append("Stroke in progress (smash/clear)")

#         # Serve legality
#         if left_elbow[1] < left_shoulder[1] and left_wrist[1] < left_elbow[1]:
#             results.append("Potential illegal serve")

#         # Court stance
#         feet_dist = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
#         if feet_dist > 150:
#             results.append("Ready stance")

#     except Exception as e:
#         results.append("Pose partially detected")
#     return results

# # ======================
# # Main Logic
# # ======================

# def process_video(input_path, output_path, model):
#     cap = cv2.VideoCapture(str(input_path))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     out = cv2.VideoWriter(str(output_path),
#                           cv2.VideoWriter_fourcc(*'mp4v'),
#                           fps,
#                           (width, height))

#     frame_count = 0
#     total_start = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.predict(frame, verbose=False)
#         for result in results:
#             if result.keypoints is None:
#                 continue

#             keypoints = result.keypoints.xy.cpu().numpy()[0]
#             confs = result.keypoints.conf.cpu().numpy()[0]
#             kp_with_conf = np.concatenate([keypoints, confs[:, None]], axis=1)

#             actions = detect_actions(kp_with_conf)
#             annotated_frame = annotate_pose_info(frame, kp_with_conf, actions)
#             out.write(annotated_frame)
#             break  # only 1 person expected

#         frame_count += 1

#     cap.release()
#     out.release()
#     print(f"[✔] Processed: {input_path.name} → {output_path.name} in {time.time() - total_start:.2f}s")


# if __name__ == "__main__":
#     input_folder = Path("input_videos")
#     output_folder = Path("annotated_videos")
#     output_folder.mkdir(exist_ok=True)

#     model = YOLO("yolov8n-pose.pt")  # Replace with 'yolov8m-pose.pt' for better accuracy

#     videos = sorted(list(input_folder.glob("*.mp4")))[:5]
#     for video in videos:
#         output_path = output_folder / f"{video.stem}_annotated.mp4"
#         process_video(video, output_path, model)

import cv2
import os
from ultralytics import YOLO
import numpy as np
import math

# Folder paths
INPUT_FOLDER = 'input_videos'
OUTPUT_FOLDER = 'output_videos'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the YOLOv8 pose model
model = YOLO('yolov8x-pose.pt')  # Ensure model file is present

def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def analyze_pose(keypoints):
    annotated_info = []
    if len(keypoints) < 33:
        return annotated_info

    try:
        left_hip, right_hip = keypoints[23], keypoints[24]
        left_knee, right_knee = keypoints[25], keypoints[26]
        left_ankle, right_ankle = keypoints[27], keypoints[28]
        left_shoulder, right_shoulder = keypoints[11], keypoints[12]
        left_elbow, right_elbow = keypoints[13], keypoints[14]
        left_wrist, right_wrist = keypoints[15], keypoints[16]

        # Stance Detection
        feet_dist = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
        if feet_dist > 150:
            annotated_info.append((int((left_ankle[0]+right_ankle[0])/2), int((left_ankle[1]+right_ankle[1])/2), 'Ready stance'))

        # Lunge Detection
        left_knee_angle = get_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
        right_knee_angle = get_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
        if left_knee_angle < 100 or right_knee_angle < 100:
            knee = left_knee if left_knee_angle < right_knee_angle else right_knee
            annotated_info.append((int(knee[0]), int(knee[1]), 'Lunge'))

        # Jump Detection (based on y-position of ankles)
        if left_ankle[1] < left_knee[1] and right_ankle[1] < right_knee[1]:
            y_pos = int(min(left_ankle[1], right_ankle[1]))
            annotated_info.append((int((left_ankle[0] + right_ankle[0]) / 2), y_pos, 'Jump in air'))

        # Stroke Mechanics
        left_elbow_angle = get_angle(left_shoulder[:2], left_elbow[:2], left_wrist[:2])
        right_elbow_angle = get_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])
        if left_elbow_angle < 100 or right_elbow_angle < 100:
            elbow = left_elbow if left_elbow_angle < right_elbow_angle else right_elbow
            annotated_info.append((int(elbow[0]), int(elbow[1]), 'Stroke in progress'))

        # Serve Legality
        if left_elbow[1] < left_shoulder[1] and left_wrist[1] < left_elbow[1]:
            annotated_info.append((int(left_elbow[0]), int(left_elbow[1]), 'Illegal serve?'))
        if right_elbow[1] < right_shoulder[1] and right_wrist[1] < right_elbow[1]:
            annotated_info.append((int(right_elbow[0]), int(right_elbow[1]), 'Illegal serve?'))

        # Hip rotation (difference in hip y-positions)
        if abs(left_hip[0] - right_hip[0]) > 50:
            center_x = int((left_hip[0] + right_hip[0]) / 2)
            center_y = int((left_hip[1] + right_hip[1]) / 2)
            annotated_info.append((center_x, center_y, 'Hip rotation'))

    except Exception:
        annotated_info.append((20, 20, 'Partial keypoints'))

    return annotated_info

def annotate_frame(frame, keypoints_list):
    for kp in keypoints_list:
        for x, y, conf in kp:
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        extra_info = analyze_pose(kp)
        for x, y, label in extra_info:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def process_video(video_path):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(OUTPUT_FOLDER, f'annotated_{video_name}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        keypoints_all = []

        for r in results:
            if r.keypoints is not None:
                for kp, conf in zip(r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()):
                    kp_with_conf = np.concatenate([kp, conf[:, None]], axis=1)
                    keypoints_all.append(kp_with_conf)

        frame = annotate_frame(frame, keypoints_all)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_path}")

def process_all_videos():
    count = 0
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith('.mp4'):
            video_path = os.path.join(INPUT_FOLDER, file)
            print(f"Processing {video_path}...")
            process_video(video_path)
            count += 1
            if count == 5:
                break

if __name__ == '__main__':
    process_all_videos()
