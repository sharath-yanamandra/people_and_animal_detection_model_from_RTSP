# People_and_animal_detection_model_from_RTSP live stream if provided with a URL:
# Person and Animal Detection using YOLOv8

## Overview

This Python script uses YOLOv8 (You Only Look Once version 8) to detect people and animals in real-time from an RTSP video stream. The application draws bounding boxes around detected objects, displays counts of detected persons and animals, and can optionally save the processed video with detections.

## Features

- Real-time person and animal detection from RTSP streams
- Visual bounding boxes with confidence scores
- On-screen counters for detected persons and animals
- Timestamp display on each frame
- Frame skipping for performance optimization
- Automatic reconnection logic for stream interruptions
- Optional video output saving

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- Ultralytics YOLOv8 (`ultralytics`)
- Numpy (`numpy`)

## Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install opencv-python ultralytics numpy
```

3. Download the YOLOv8 model weights (or use your own trained model):
   - You can download pretrained models from the Ultralytics GitHub repository
   - Place the model file (e.g., `yolov8x.pt`) in your project directory

## Usage

1. Edit the `main()` function in `people_and_animals.py`:
   - Replace the `rtsp_url` with your actual RTSP stream URL
   - Optionally modify the output path for saved videos

2. Run the script:

```bash
python people_and_animals.py
```

3. While running:
   - Press 'q' to quit the application
   - Detections will be displayed in real-time
   - If an output path was specified, the processed video will be saved

## Configuration Options

- **Model path**: Change `'path/to/your/yolov8x.pt'` to your actual model file path
- **Frame skipping**: Adjust `frame_skip` value to process fewer frames for better performance
- **Reconnection attempts**: Modify `max_reconnect_attempts` to change how many times the script tries to reconnect
- **Colors**: Customize bounding box colors in the `colors` dictionary
- **Class IDs**: Adjust `person_class_id` and `animal_class_ids` if using a custom model

## Notes

- The script uses COCO dataset class IDs by default (person=0, animals=16-22)
- For best performance, use a GPU-enabled environment
- RTSP streams can be unstable - the script includes reconnection logic to handle interruptions
- Frame skipping helps with performance but reduces detection frequency

## Troubleshooting

- **Connection issues**: Verify your RTSP URL and network connectivity
- **Model loading errors**: Ensure the model file path is correct and the file exists
- **Performance problems**: Increase the `frame_skip` value or reduce the input resolution
- **Missing dependencies**: Make sure all required packages are installed

## Output Example

The script will display a window with:
- Bounding boxes around detected persons (green) and animals (blue)
- Counters showing the number of detected persons and animals
- Current timestamp in the bottom of the frame
- If specified, a video file will be saved with all detections
