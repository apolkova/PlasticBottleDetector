import csv
import json
import time
from datetime import datetime
from pathlib import Path
import threading

import cv2
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox

from ultralytics import YOLO


# CONFIG
MODEL_PATH = r"..\models\model_1\weights\best.pt"    
IMAGE_PATH = r"path_to_image" 

OUTPUT_DIR = Path(r"..\app_output") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Banchmark latency

model = YOLO(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

def benchmark(device, runs=30, warmup=5):
    times = []

    # Warm-up runs are not counted
    for _ in range(warmup):
        model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            iou=0.70,
            device=device,
            verbose=False
        )

    for _ in range(runs):
        start = time.perf_counter()

        model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            iou=0.70,
            device=device,
            verbose=False
        )

        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"{device}:")
    print(f"  Average latency: {avg_ms:.2f} ms")
    print(f"  Min latency: {min_ms:.2f} ms")
    print(f"  Max latency: {max_ms:.2f} ms")
    print(f"  Approx FPS: {1000 / avg_ms:.2f}")

benchmark("cpu")

# APP
class BottleQCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plastic Bottle Quality Control Dashboard")
        self.root.geometry("1700x850")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.model = YOLO(MODEL_PATH)

        self.class_names = {
            0: "bottle",
            1: "cap",
            2: "label",
            3: "liquid",
        }

        self.target_class_ids = {
            "bottle": 0,
            "cap": 1,
            "label": 2,
            "liquid": 3,
        }

        self.class_conf_thresholds = {
            "bottle": 0.7,
            "cap": 0.64,
            "label": 0.62,
            "liquid": 0.62,
        }

        self.class_colors = {
            "bottle": (255, 127, 0),    # RGB (0, 127, 255)
            "cap":    (240, 207, 137),  # RGB (137, 207, 240)
            "label":  (112, 25, 25),    # RGB (25, 25, 112)
            "liquid": (255, 255, 0),    # RGB (0, 255, 255)
        }

        self.video_running = False
        self.cap = None

        self.total_inspections = 0
        self.total_detections = 0
        self.failed_inspections = 0
        self.inspection_log_rows = []

        self.current_annotated_image = None

        self.build_ui()

    # -----------------------------------------------------------------
    # System messages go to terminal, not inspection log
    # -----------------------------------------------------------------
    def system_message(self, message):
        print(message)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def build_ui(self):
        self.root.grid_columnconfigure(0, weight=1)  # left settings panel
        self.root.grid_columnconfigure(1, weight=3)  # center image panel
        self.root.grid_columnconfigure(2, weight=1)  # right status/log panel
        self.root.grid_rowconfigure(0, weight=1)

        # ================================================================
        # LEFT PANEL — SETTINGS + DETECTED OBJECTS
        # ================================================================
        self.left_panel = ctk.CTkScrollableFrame(self.root, width=320)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(
            self.left_panel,
            text="Detection Controls",
            font=("Arial", 22, "bold"),
        ).pack(pady=(10, 15))

        # Settings box
        self.settings_box = ctk.CTkFrame(self.left_panel)
        self.settings_box.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            self.settings_box,
            text="Per-Class Confidence",
            font=("Arial", 16, "bold"),
        ).pack(pady=(15, 5))

        self.class_conf_sliders = {}
        self.class_conf_labels = {}

        for class_name, default_value in self.class_conf_thresholds.items():
            label = ctk.CTkLabel(
                self.settings_box,
                text=f"{class_name.capitalize()}: {default_value:.2f}",
            )
            label.pack(pady=(5, 0))

            slider = ctk.CTkSlider(
                self.settings_box,
                from_=0.05,
                to=0.95,
                number_of_steps=90,
                command=lambda value, name=class_name: self.update_class_conf_label(name, value),
            )
            slider.set(default_value)
            slider.pack(fill="x", padx=15, pady=5)

            self.class_conf_labels[class_name] = label
            self.class_conf_sliders[class_name] = slider

        ctk.CTkLabel(
            self.settings_box,
            text="NMS / Overlap",
            font=("Arial", 16, "bold"),
        ).pack(pady=(20, 5))

        self.iou_label = ctk.CTkLabel(self.settings_box, text="IoU: 0.70")
        self.iou_label.pack()

        self.iou_slider = ctk.CTkSlider(
            self.settings_box,
            from_=0.10,
            to=0.95,
            number_of_steps=85,
            command=self.update_iou_label,
        )
        self.iou_slider.set(0.70)
        self.iou_slider.pack(fill="x", padx=15, pady=5)

        self.imgsz_label = ctk.CTkLabel(self.settings_box, text="Image Size")
        self.imgsz_label.pack(pady=(15, 0))

        self.imgsz_option = ctk.CTkOptionMenu(
            self.settings_box,
            values=["416", "512", "640", "800"],
        )
        self.imgsz_option.set("640")
        self.imgsz_option.pack(pady=5)

        self.device_option = ctk.CTkOptionMenu(
            self.settings_box,
            values=["cpu"],
        )
        self.device_option.set("cpu")
        self.device_option.pack(pady=10)

        # Detected objects box
        ctk.CTkLabel(
            self.left_panel,
            text="Detected Objects",
            font=("Arial", 18, "bold"),
        ).pack(pady=(20, 5))

        self.detection_box = ctk.CTkTextbox(self.left_panel, height=220)
        self.detection_box.pack(fill="x", padx=10, pady=10)
        self.detection_box.insert("end", "Detected objects will appear here.\n")

        # ================================================================
        # CENTER PANEL — IMAGE / VIDEO DISPLAY + BUTTONS
        # ================================================================
        self.center_panel = ctk.CTkFrame(self.root)
        self.center_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.center_panel.grid_rowconfigure(1, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.center_panel,
            text="Plastic Bottle Quality Control",
            font=("Arial", 28, "bold"),
        )
        self.title_label.grid(row=0, column=0, pady=15)

        self.image_label = ctk.CTkLabel(
            self.center_panel,
            text="Upload an image, upload a video, or start live feed",
            width=780,
            height=520,
        )
        self.image_label.grid(row=1, column=0, padx=15, pady=15, sticky="nsew")

        self.button_frame = ctk.CTkFrame(self.center_panel)
        self.button_frame.grid(row=2, column=0, pady=10)

        self.upload_image_btn = ctk.CTkButton(
            self.button_frame,
            text="Upload Picture",
            command=self.upload_picture,
        )
        self.upload_image_btn.grid(row=0, column=0, padx=8, pady=8)

        self.upload_video_btn = ctk.CTkButton(
            self.button_frame,
            text="Upload Video",
            command=self.upload_video,
        )
        self.upload_video_btn.grid(row=0, column=1, padx=8, pady=8)

        self.live_btn = ctk.CTkButton(
            self.button_frame,
            text="Start Live Feed",
            command=self.start_live_feed,
        )
        self.live_btn.grid(row=0, column=2, padx=8, pady=8)

        self.stop_btn = ctk.CTkButton(
            self.button_frame,
            text="Stop",
            command=self.stop_video,
        )
        self.stop_btn.grid(row=0, column=3, padx=8, pady=8)

        self.save_btn = ctk.CTkButton(
            self.button_frame,
            text="Save Screenshot",
            command=self.save_screenshot,
        )
        self.save_btn.grid(row=0, column=4, padx=8, pady=8)

        # ================================================================
        # RIGHT PANEL — QC STATUS + COUNTERS + LOG
        # ================================================================
        self.right_panel = ctk.CTkScrollableFrame(self.root, width=360)
        self.right_panel.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.status_label = ctk.CTkLabel(
            self.right_panel,
            text="WAITING",
            font=("Arial", 24, "bold"),
        )
        self.status_label.pack(pady=(10, 15))

        ctk.CTkLabel(
            self.right_panel,
            text="Quality-Control Status",
            font=("Arial", 18, "bold"),
        ).pack(pady=(10, 5))

        self.qc_box = ctk.CTkTextbox(self.right_panel, height=170)
        self.qc_box.pack(fill="x", padx=10, pady=10)
        self.qc_box.insert("end", "Quality-control status will appear here.\n")

        ctk.CTkLabel(
            self.right_panel,
            text="Inspection Counters",
            font=("Arial", 18, "bold"),
        ).pack(pady=(15, 5))

        self.counter_box = ctk.CTkTextbox(self.right_panel, height=110)
        self.counter_box.pack(fill="x", padx=10, pady=10)
        self.update_counters()

        ctk.CTkLabel(
            self.right_panel,
            text="Inspection Log",
            font=("Arial", 18, "bold"),
        ).pack(pady=(15, 5))

        self.log_box = ctk.CTkTextbox(self.right_panel, height=250)
        self.log_box.pack(fill="x", padx=10, pady=10)

        self.log_button_frame = ctk.CTkFrame(self.right_panel)
        self.log_button_frame.pack(fill="x", padx=10, pady=5)

        self.reset_log_btn = ctk.CTkButton(
            self.log_button_frame,
            text="Reset Log",
            command=self.reset_log,
        )
        self.reset_log_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.download_log_btn = ctk.CTkButton(
            self.log_button_frame,
            text="Download CSV",
            command=self.download_log,
        )
        self.download_log_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.log_button_frame.grid_columnconfigure(0, weight=1)
        self.log_button_frame.grid_columnconfigure(1, weight=1)

    # -----------------------------------------------------------------
    # UI helpers
    # -----------------------------------------------------------------
    def update_iou_label(self, value):
        self.iou_label.configure(text=f"IoU: {float(value):.2f}")

    def update_class_conf_label(self, class_name, value):
        self.class_conf_labels[class_name].configure(
            text=f"{class_name.capitalize()}: {float(value):.2f}"
        )

    def reset_log(self):
        self.log_box.delete("1.0", "end")
        self.inspection_log_rows = []

    def download_log(self):
        if not self.inspection_log_rows:
            messagebox.showwarning("No Log Data", "No inspection results to export.")
            return

        csv_path = OUTPUT_DIR / f"inspection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        fieldnames = [
            "timestamp",
            "source",
            "status",
            "bottle_count",
            "cap_count",
            "label_count",
            "liquid_count",
            "total_detections",
            "max_confidence",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.inspection_log_rows)

        messagebox.showinfo("CSV Saved", f"Inspection log saved to:\n{csv_path}")
        self.system_message(f"CSV saved: {csv_path}")

    def update_counters(self):
        self.counter_box.delete("1.0", "end")
        self.counter_box.insert("end", f"Images/Frames inspected: {self.total_inspections}\n")
        self.counter_box.insert("end", f"Total detections: {self.total_detections}\n")
        self.counter_box.insert("end", f"Failed inspections: {self.failed_inspections}\n")

    def log_detection(self, source_name, detections, status):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        bottle_count = sum(1 for d in detections if d["class_name"] == "bottle")
        cap_count = sum(1 for d in detections if d["class_name"] == "cap")
        label_count = sum(1 for d in detections if d["class_name"] == "label")
        liquid_count = sum(1 for d in detections if d["class_name"] == "liquid")

        total_detections = len(detections)

        if detections:
            max_confidence = max(d["confidence"] for d in detections)
        else:
            max_confidence = 0.0

        row = {
            "timestamp": timestamp,
            "source": source_name,
            "status": status,
            "bottle_count": bottle_count,
            "cap_count": cap_count,
            "label_count": label_count,
            "liquid_count": liquid_count,
            "total_detections": total_detections,
            "max_confidence": round(max_confidence, 4),
        }

        self.inspection_log_rows.append(row)

        self.log_box.insert("end", f"[{timestamp}] {source_name}\n")
        self.log_box.insert("end", f"Status: {status}\n")
        self.log_box.insert(
            "end",
            f"Detected: bottle={bottle_count}, cap={cap_count}, "
            f"label={label_count}, liquid={liquid_count}\n",
        )
        self.log_box.insert("end", f"Total detections: {total_detections}\n")
        self.log_box.insert("end", f"Max confidence: {max_confidence:.2f}\n")
        self.log_box.insert("end", "-" * 55 + "\n")
        self.log_box.see("end")

    def rgb_to_bgr(self, rgb):
        r, g, b = rgb
        return (r, g, b)

    # -----------------------------------------------------------------
    # Input methods
    # -----------------------------------------------------------------
    def upload_picture(self):
        self.stop_video()

        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.system_message(f"Image uploaded: {Path(file_path).name}")
            self.process_image(file_path, save_outputs=True)

    def upload_video(self):
        self.stop_video()

        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.system_message(f"Video uploaded: {Path(file_path).name}")
            self.video_running = True
            thread = threading.Thread(target=self.process_video, args=(file_path,))
            thread.daemon = True
            thread.start()

    def start_live_feed(self):
        self.stop_video()

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.system_message("ERROR: Could not open webcam.")
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return

        self.video_running = True
        self.process_live_frame()

    def stop_video(self):
        self.video_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None


    # Processing
    def process_image(self, image_path, save_outputs=True):
        frame = cv2.imread(str(image_path))

        if frame is None:
            self.system_message(f"ERROR: Could not read image: {image_path}")
            messagebox.showerror("Image Error", "Could not read selected image.")
            return

        annotated, detections = self.run_detection(frame)

        self.current_annotated_image = annotated
        self.display_frame(annotated)

        status = self.display_results(detections)
        self.log_detection(Path(image_path).name, detections, status)

        self.total_inspections += 1
        self.total_detections += len(detections)

        if "FAIL" in status:
            self.failed_inspections += 1

        self.update_counters()

        if save_outputs:
            self.save_detection_outputs(image_path, annotated, detections)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.system_message("ERROR: Could not open video.")
            messagebox.showerror("Video Error", "Could not open selected video.")
            return

        frame_count = 0

        while self.video_running:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Process every 5th frame for speed
            if frame_count % 5 != 0:
                continue

            annotated, detections = self.run_detection(frame)
            self.current_annotated_image = annotated

            self.root.after(0, self.display_frame, annotated)

            status = self.display_results(detections)

            #loging status to logs
            self.log_detection(f"Video frame {frame_count}", detections, status)

            self.total_inspections += 1
            self.total_detections += len(detections)

            if "FAIL" in status:
                self.failed_inspections += 1

            self.root.after(0, self.update_counters)

            time.sleep(0.03)

        cap.release()
        self.video_running = False
        self.system_message("Video processing finished.")

    def process_live_frame(self):
        if not self.video_running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if ret:
            annotated, detections = self.run_detection(frame)
            self.current_annotated_image = annotated

            self.display_frame(annotated)

            status = self.display_results(detections)

            # For live feed, loging status
            self.log_detection("Live camera frame", detections, status)

            self.total_inspections += 1
            self.total_detections += len(detections)

            if "FAIL" in status:
                self.failed_inspections += 1

            self.update_counters()

        self.root.after(100, self.process_live_frame)

    def run_detection(self, frame):
        iou = float(self.iou_slider.get())
        imgsz = int(self.imgsz_option.get())
        device = self.device_option.get()

        results = self.model.predict(
            source=frame,
            conf=0.05,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf)

            class_slider = self.class_conf_sliders.get(class_name)

            if class_slider is not None:
                required_confidence = float(class_slider.get())
            else:
                required_confidence = 0.25

            if confidence < required_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "threshold_used": required_confidence,
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                }
            })

        annotated = frame.copy()

        h, w = annotated.shape[:2]
        scale_factor = max(w, h) / 1000.0

        box_thickness = max(4, int(5 * scale_factor))
        font_scale = max(0.9, 1.0 * scale_factor)
        text_thickness = max(2, int(3 * scale_factor))
        padding = max(8, int(10 * scale_factor))

        for det in detections:
            bbox = det["bbox"]

            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

            class_name = det["class_name"]
            confidence = det["confidence"]
            label = f"{class_name} {confidence:.2f}"

            color_rgb = self.class_colors.get(class_name, (0, 127, 255))
            color = self.rgb_to_bgr(color_rgb)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                box_thickness,
            )

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )

            label_top = max(y1 - text_height - padding - 4, 0)
            label_bottom = y1
            label_right = x1 + text_width + padding * 2

            cv2.rectangle(
                annotated,
                (x1, label_top),
                (label_right, label_bottom),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1 + padding, max(y1 - 6, text_height + 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        return annotated, detections

    # -----------------------------------------------------------------
    # Display / save
    # -----------------------------------------------------------------
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((780, 520))

        ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )

        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image

    def display_results(self, detections):
        self.detection_box.delete("1.0", "end")
        self.qc_box.delete("1.0", "end")

        detected_classes = [det["class_name"] for det in detections]

        has_bottle = "bottle" in detected_classes
        has_cap = "cap" in detected_classes
        has_label = "label" in detected_classes
        has_liquid = "liquid" in detected_classes

        #failed and pass
        if has_bottle and has_cap and has_label and has_liquid:
            final_status = "PASS"
            self.status_label.configure(text="PASS", text_color="green")
        else:
            final_status = "FAILED"
            self.status_label.configure(text="FAILED", text_color="red")

        if not detections:
            self.detection_box.insert("end", "No objects detected.\n")
        else:
            self.detection_box.insert("end", f"Detected objects: {len(detections)}\n\n")

            for i, det in enumerate(detections, start=1):
                self.detection_box.insert(
                    "end",
                    f"{i}. {det['class_name']} "
                    f"- confidence: {det['confidence']:.2f} "
                    f"- threshold: {det['threshold_used']:.2f}\n",
                )

        self.qc_box.insert("end", "Quality Control Result:\n")
        self.qc_box.insert("end", f"- Bottle presence: {'PASS' if has_bottle else 'FAIL'}\n")
        self.qc_box.insert("end", f"- Cap presence: {'PASS' if has_cap else 'FAIL'}\n")
        self.qc_box.insert("end", f"- Label presence: {'PASS' if has_label else 'FAIL'}\n")
        self.qc_box.insert("end", f"- Liquid presence: {'PASS' if has_liquid else 'FAIL'}\n")
        self.qc_box.insert("end", f"\nFinal decision: {final_status}\n")

        return final_status

    def save_detection_outputs(self, image_path, annotated, detections):
        image_path = Path(image_path)

        output_image_path = OUTPUT_DIR / f"{image_path.stem}_detected.jpg"
        output_json_path = OUTPUT_DIR / f"{image_path.stem}_detections.json"

        cv2.imwrite(str(output_image_path), annotated)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

        self.system_message(f"Saved annotated image: {output_image_path}")
        self.system_message(f"Saved JSON: {output_json_path}")

    def save_screenshot(self):
        if self.current_annotated_image is None:
            messagebox.showwarning("No Image", "No annotated image available to save.")
            return

        screenshot_path = OUTPUT_DIR / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(screenshot_path), self.current_annotated_image)

        self.system_message(f"Screenshot saved: {screenshot_path}")
        messagebox.showinfo("Saved", f"Screenshot saved to:\n{screenshot_path}")


# ---------------------------------------------------------------------
# RUN APP
# ---------------------------------------------------------------------
if __name__ == "__main__":
    root = ctk.CTk()
    app = BottleQCApp(root)
    root.mainloop()