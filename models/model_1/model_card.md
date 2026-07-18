# Model Card — model_version4_used (v4.0)

## 1. Overview

- **Goal:** Detect whether a plastic bottle has all required parts for a basic quality-control check.
- **Task:** Object Detection
- **Classes:** bottle, cap, label, liquid
- **Intended users:** QA operators, teachers/evaluators, developers, and students testing the demo application.

The model is used to support a simple inspection workflow. A bottle passes only when the model detects the bottle body, cap, label, and visible liquid. If one or more required parts are missing, the application marks the inspection as FAILED.

## 2. Intended Use and Scope

- **Intended environment:** Prototype plastic bottle quality-control setup, for example a bottle on a table, inspection area, or production-line/conveyor-belt scenario.
- **Assumptions:** The bottle is visible, mostly upright, and photographed from a normal camera view. The model works best with enough lighting, limited occlusion, and a background that does not hide the bottle parts.
- **Out-of-scope uses:** The model is not designed to check exact liquid volume, cap tightness, barcode correctness, label text, expiration date, bottle damage, scratches, product type, contamination, or liquid quality.

The model only checks whether the required classes are detected. It does not measure if the liquid level is correct.

## 3. Training Data

- **Data source:** The dataset was collected by the project team. Images were taken in grocery stores, at home, and from older gallery photos. Extra images were created at home using colored water to make the liquid class more diverse.
- **Devices used:** iPhone 16 and Motorola Moto G24 Power.
- **Dataset size:** The original dataset contained 680 images.

Original split before augmentation:

| Split         | Percentage    | Number of images  |
|---            |---:           |---:               |          
| Train         | 70%           | 476               |
| Validation    | 20%           | 136               |
| Test          | 10%           | 68                |

After Roboflow augmentation, the exported dataset showed a different percentage split because augmented images were added only to the training set. Validation and test images were not augmented.

- **Class distribution:** The dataset contains four annotation classes: bottle, cap, label, and liquid. The exact per-class annotation counts should be taken from the Roboflow dataset statistics or YOLO label plot.
- **Labeling guidelines:** Bounding boxes were created for visible bottle parts only. If a part was missing or not visible, it was not labeled. For example, if a bottle had no visible cap, the cap class was not annotated.
- **Edge cases:** Difficult cases included transparent bottles, reflective surfaces, small caps, partially visible labels, and liquid regions that were hard to separate from the bottle body.
- **Augmentations:** Roboflow augmentation was used to increase dataset diversity.

Roboflow augmentation settings:

| Augmentation                  | Setting                   |
|---                            |---                        |
| Outputs per training example  | 3                         |
| Grayscale                     | Applied to 15% of images  |
| Hue                           | Between -40° and +40°     |
| Saturation                    | Between -25% and +25%     |
| Exposure                      | Between -12% and +12%     |

These augmentations were used because bottles can appear in different lighting conditions, colors, and backgrounds. Rotation augmentation was not used because the target use case assumes bottles are inspected upright.

## 4. Evaluation

- **Test set description:** The test set contains 68 original images separated from the training set before augmentation. Test images were not augmented, so they represent unseen examples for evaluation.
- **Metrics:** The model was evaluated using standard YOLOv8 object detection metrics.

Overall metrics:

| Metric        | Value |
|---            |---:   |  
| Precision     | 0.629 |
| Recall        | 0.560 |
| mAP@0.5       | 0.545 |
| mAP@0.5:0.95  | 0.349 |

The best validation result was reached at epoch 19.

- **Per-class analysis:** Bottle detection was the most stable because the bottle is usually the largest and most visible object. Cap detection was harder because caps are small and can be partly hidden. Label detection worked best when the label was clearly visible, but reflections and curved surfaces made it harder. Liquid detection was difficult because liquid can be transparent, dark, reflective, or visually similar to the bottle body.
- **Qualitative analysis:** Successful examples usually had a clearly visible upright bottle with visible cap, label, and liquid. Failure examples included missed caps, missed liquid, double liquid detections, reflections on labels, and cluttered backgrounds.
- **Recommended confidence thresholds:**

| Class     | Confidence threshold  |
|---        |---:                   |
| bottle    | 0.70                  |
| cap       | 0.64                  |
| label     | 0.62                  |
| liquid    | 0.62                  |

The bottle threshold is higher because the bottle is usually easier to detect. The other thresholds were adjusted during testing to reduce false detections while keeping the app usable.

## 5. Limitations and Failure Modes

Known failure conditions:

| Failure mode                          | Typical reason                                                                    |
|---                                    |---                                                                                |  
| Cap not detected                      | Cap is small, hidden, or similar to the background                                |
| Liquid not detected                   | Liquid is transparent, dark, reflective, or hard to separate from the bottle      |
| Double liquid detection               | Label or reflection may be confused with liquid                                   |
| Label not detected                    | Glare, reflection, curved shape, or partial visibility                            |
| False positives in cluttered images   | Background objects may look similar to bottle parts                               |
| Missing detections in partial views   | Some bottle parts are outside the camera view                                     |

Typical false negatives happen with small caps, transparent liquid, dark bottles, and partially visible objects. Typical false positives happen when reflections, labels, or background objects look similar to the target classes.

Suggested improvements include collecting more images of small caps, transparent liquids, reflective labels, and factory-like backgrounds. A fixed camera position and stable lighting would also improve reliability.

## 6. Deployment Notes

- **Input requirements:** Standard RGB images or video frames. The model was trained with image size 640. For best results, the bottle should be visible, upright, and not heavily occluded.
- **Preprocessing:** Images are resized by YOLOv8 during inference. No special manual preprocessing is required in the demo app.
- **Output format:** The model returns bounding boxes, class IDs, class names, and confidence scores. Bounding boxes use the YOLO/OpenCV convention with coordinates in the image space: x1, y1, x2, y2.
- **Score meaning:** The confidence score shows how confident the model is that a detected bounding box belongs to a class.
- **Model file:** `best.pt`
- **Model size:** 49.6 MB / 52,039,890 bytes
- **Post-processing:** Non-Maximum Suppression is used to reduce duplicate overlapping detections.
- **Recommended IoU/NMS threshold:** 0.70
- **Compute:** The model was trained on a Dell Pro Max 16 notebook with 32 GB RAM, Intel Arc Pro 140T GPU, and Intel Core Ultra 7 255H 2.00 GHz processor. Training took approximately 26 hours.
- **Approximate latency:** During testing, image inference usually took less than 2 seconds. Video and live camera inference ran at approximately 3-5 FPS, depending on input resolution, hardware, and number of detections.

The demo application was built in Python using CustomTkinter. It supports image upload, video upload, live camera feed, bounding boxes, class labels, confidence values, PASS/FAILED status, confidence sliders, IoU/NMS setting, inspection counters, and CSV log export.

## 7. Ethical / Safety / Privacy Considerations

The dataset focuses on plastic bottles and bottle parts. No personal identification is needed for the model. If people or sensitive information appear in images, they should be avoided, cropped out, or not used in the dataset.

The model should not be used as the only decision system in a real factory without additional validation. False positives and false negatives are possible, especially with poor lighting, reflections, cluttered backgrounds, or unusual bottle shapes.

## 8. Versioning and Contact

- **Version:** v4.0
- **Date:** 2026
- **Authors:** Adéla Polková
- **Contact:** a_polkova@utb.cz, 8250919@estg.ipp.pt
