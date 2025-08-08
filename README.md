# Badminton-Shot-Mechanics-Movement-Profiling

#  Badminton Player Pose Analysis & Annotation (YOLOv8-Pose)

This project uses **YOLOv8-Pose** to detect and analyze badminton players' biomechanics and annotate key performance metrics directly onto the video frames. It supports **multi-player analysis** and outputs annotated videos for easy visualization.

---

##  Folder Structure

```
├── input_videos/           # Place your raw badminton match videos here (up to 5)
├── output_videos/          # Annotated videos will be saved here
├── badminton_analysis.py   # Main script
├── README.md               # This file
```

---

##  Key Features Analyzed

###  Stance & Court Movement

* Readiness stance
* Lunge detection
* Split-step & footwork patterns

###  Stroke Mechanics

* Elbow and wrist angles
* Smash/Clear posture analysis
* Joint coordination insights

###  Shot-specific Posture

* Jump and landing detection
* Hip rotation
* Follow-through angles

###  Serve Legality

* Elbow/wrist/shoulder positioning
* Illegal serve warning (e.g., wrist above shoulder)

### Reaction Timing

* Movement latency post opponent’s stroke
* Anticipation patterns & recovery tracking

---

##  Requirements

* Python ≥ 3.8
* [Ultralytics](https://docs.ultralytics.com/) YOLOv8
* OpenCV
* NumPy

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

##  How to Run

1. Place up to **5 .mp4 videos** in the `input_videos/` folder.
2. Run the analysis script:

```bash
python badminton_analysis.py
```

3. Check the `output_videos/` folder for annotated results.

---

##  Model Used

* **YOLOv8x-pose** (`yolov8x-pose.pt`) – High accuracy pose estimation model from Ultralytics.
* You can replace with `yolov8n-pose.pt` or `yolov8m-pose.pt` for speed over accuracy.

---
##  Output

The output videos are fully annotated with pose-based visualizations using YOLOv8-Pose. Each player in the video is tracked and analyzed individually, and key biomechanical events are overlaid directly on the video.

Included Visual Elements:
 Skeleton keypoints for all visible players

 Real-time annotations for:

Stance & Footwork (e.g., Ready stance, Lunge)

Stroke Mechanics (e.g., Elbow angle, Contact point posture)

Serve Legality (e.g., Possible Illegal Serve)

Reaction Timing (e.g., Movement latency post shot)

Player-specific multi-person analysis with no need for manual labeling

Each output video is saved to the output_videos/ folder, named as annotated_<input_video_name>.mp4.

---

## Key Insights Enabled:


Players' movement efficiency, readiness, and balance

Stroke quality based on elbow/wrist/shoulder coordination

Serve legality errors in doubles

Reaction time from opponent actions

Ability to detect both offensive and defensive posture changes


---

github: https://github.com/soniya-naskar
linkedin: https://www.linkedin.com/in/soniya-naskar-8279801a3/
