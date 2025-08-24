from ultralytics import YOLO
import cv2
import os
import time
import csv

last_saved_time = 0  # Global variable to track save intervals

def check_model(model_path):
    """Checks if the model file exists and loads it."""
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' does not exist.")
        return None
    try:
        model = YOLO(model_path)  # Load the YOLO model
        print(f"Successfully loaded model from '{model_path}'")
        return model
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return None

def save_detection(frame, output_dir="detections", image_name="frame"):
    """Saves the detected frame as an image in the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"{image_name}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✅ Saved detection frame: {filename}")

def process_frame(model, frame, image_name="unknown", save_interval=3, log_writer=None):
    """Runs YOLO detection on a single frame and draws bounding boxes + logs performance."""
    global last_saved_time

    start_time = time.time()
    results = model.predict(frame, verbose=False)  # Inference
    inference_time = time.time() - start_time
    
    detection_found = False
    detections_info = []
    
    for result in results:
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                label = int(box.cls[0].item())
                
                # Draw bounding box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                label_text = f"Class: {label} ({conf:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                detections_info.append((label, conf))
                detection_found = True
    
    # Print results
    print(f"\nImage: {image_name}")
    print(f"Inference time: {inference_time:.3f} sec")
    if detections_info:
        print(f"Detections ({len(detections_info)}): {detections_info}")
    else:
        print("No detections found.")

    # Save detection image if found
    if detection_found and (time.time() - last_saved_time >= save_interval):
        save_detection(frame, image_name=image_name)
        last_saved_time = time.time()

    # Log results in CSV
    if log_writer:
        if detections_info:
            for (label, conf) in detections_info:
                log_writer.writerow([image_name, inference_time, label, conf])
        else:
            log_writer.writerow([image_name, inference_time, "None", 0.0])

    return frame

def test_image(model, image_path, log_writer, max_width=800, max_height=600):
    """Runs the YOLO model on a single image and logs results."""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
    
    frame = cv2.imread(image_path)
    frame = process_frame(model, frame, image_name=os.path.basename(image_path), log_writer=log_writer)
    
    # Resize if too large
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    cv2.imshow("YOLO Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_folder(model, folder_path):
    """Processes all image and video files in a folder and logs results."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    valid_image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    valid_video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

    files = os.listdir(folder_path)
    files.sort()

    with open("results.csv", "w", newline="") as f:
        log_writer = csv.writer(f)
        log_writer.writerow(["Image/Video", "InferenceTime(s)", "Class", "Confidence"])

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            ext = os.path.splitext(file_name)[1].lower()

            if ext in valid_image_ext:
                print(f"\nProcessing image: {file_name}")
                test_image(model, file_path, log_writer)
            elif ext in valid_video_ext:
                print(f"\nSkipping video logging (only shows detections): {file_name}")
            else:
                print(f"Skipping unsupported file: {file_name}")

if __name__ == "__main__":
    model_path = r"model\\Combined_Best.pt"
    model = check_model(model_path)
    
    if model:
        folder_name = input("Enter folder name containing images/videos: ").strip()
        if not os.path.isabs(folder_name):
            folder_path = os.path.join(os.getcwd(), folder_name)
        else:
            folder_path = folder_name
        process_folder(model, folder_path)
