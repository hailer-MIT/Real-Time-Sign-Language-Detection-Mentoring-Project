# Real-Time Sign Language Detection Using TensorFlow Object Detection API

##  Problem Statement

For individuals with speech or hearing impairments, sign language is a vital communication tool. However, the lack of widespread understanding of sign language creates a communication barrier. A real-time translator powered by AI can help close that gap.

## Solution

This project builds a real-time sign language detection system using **TensorFlow 2 Object Detection API**. By training a custom object detector with transfer learning, it recognizes 5 static American Sign Language (ASL) signs — "Hello", "Thanks", "Yes", "No", and "Sorry" — using webcam input.

## Approach

The project involves the complete end-to-end pipeline:
-  **Image Collection**: Webcam-based image capture using OpenCV (2,500+ images total)
-  **Annotation**: Each sign is labeled using LabelImg and converted into TFRecord format
-  **Transfer Learning**: SSD MobileNetV2 pre-trained model fine-tuned on the custom dataset
-  **Detection**: Real-time inference via webcam with bounding boxes and confidence scores

## Tools & Technologies

- TensorFlow 2.0 Object Detection API
- Python
- OpenCV
- LabelImg (for annotation)
- SSD MobileNet V2 (pretrained on COCO dataset)

## Results

- Real-time detection speed: ~15 FPS
- Detection Recall: ~89% at 50% confidence threshold
- Lightweight and responsive even on standard GPUs

## Dataset

Custom dataset collected using webcam and annotated manually.
Classes:
- hello
- thanks
- yes
- no
- sorry

Dataset size: ~2,500 images  
Not included in this repo due to size. You can generate your own using the included scripts.
