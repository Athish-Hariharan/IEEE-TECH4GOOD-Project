from ultralytics import YOLO
import cv2
import os
import time

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

def save_detection(frame, output_dir="detections"):
    """Saves the detected frame as an image in the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"detection_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved detection frame: {filename}")

def process_frame(model, frame, save_interval=3):
    """Runs YOLO detection on a single frame and draws bounding boxes."""
    last_saved_time = time.time()
    results = model.predict(frame)
    detection_found = False
    
    for result in results:
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to int
                conf = box.conf[0].item()  # Convert tensor to float
                label = int(box.cls[0].item())  # Convert tensor to int
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                label_text = f"Class: {label} ({conf:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detection_found = True
    
    if detection_found and (time.time() - last_saved_time >= save_interval):
        save_detection(frame)
    
    return frame

def test_video(model, video_source):
    """Runs the YOLO model on a video file."""
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_source}'.")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"Finished processing video: {video_source}")
            break
        
        frame = process_frame(model, frame)
        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_image(model, image_path, max_width=800, max_height=600):
    """Runs the YOLO model on a single image."""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return
    
    frame = cv2.imread(image_path)
    frame = process_frame(model, frame)
    
    # Resize if too large
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    cv2.imshow("YOLO Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_folder(model, folder_path):
    """Processes all image and video files in a folder."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    valid_image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    valid_video_ext = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

    files = os.listdir(folder_path)
    files.sort()  # Process in order

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        ext = os.path.splitext(file_name)[1].lower()

        if ext in valid_image_ext:
            print(f"Processing image: {file_name}")
            test_image(model, file_path)
        elif ext in valid_video_ext:
            print(f"Processing video: {file_name}")
            test_video(model, file_path)
        else:
            print(f"Skipping unsupported file: {file_name}")

if __name__ == "__main__":
    model_path = r"model\\Combined_Best.pt"  # Change as needed
    model = check_model(model_path)
    
    if model:
        folder_name = input("Enter folder name containing images/videos: ").strip()
        
        # If user only gives a name, assume it's in the same directory as the script
        if not os.path.isabs(folder_name):
            folder_path = os.path.join(os.getcwd(), folder_name)
        else:
            folder_path = folder_name
        
        process_folder(model, folder_path)