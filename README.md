# 🚗 Vehicle Detection with YOLOv3 for Traffic Management

This repository contains a Python implementation of vehicle detection using YOLOv3. It is part of a research project focused on analyzing the impact of object detection models, such as YOLOv3, on vehicle management.

---

## Project Objective

This research aimed to analyze the impact of deep learning based vehicle detection. YOLOv3 was selected for its performance in object detection tasks with minimal computational cost.

---

## Project Contents

| File                          | Description                                   |
|-------------------------------|-----------------------------------------------|
| `yolov3_vehicle_detection.py` | Main Python script for vehicle detection      |
| `video_name.mp4`              | Sample input video (not provided here)        |

> **Note:** Model files are **not included** in this repository for copyright and file size reasons. Instructions to download them are provided below.

---

## Required Files (Download Manually)

To run this project, you will need the following files manually downloaded:

### 1. YOLOv3 Weights

Download from the official YOLO site:  
[all files](https://pjreddie.com/darknet/yolo/)

Save them as:  
`yolov3.weights`
`yolov3.cfg`
`coco.names`

---

> Make sure all three files are placed in the **same folder** as the Python script.

---

## Classes Detected

The model is filtered to detect only the following vehicle-related classes:

- `car`
- `bus`
- `truck`
- `motorbike`
- `bicycle`

---

