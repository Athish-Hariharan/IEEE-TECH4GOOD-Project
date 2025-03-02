# -- coding: utf-8 --
"""
Color Video Frame Extractor

Features:
1. Uploads a video file.
2. Extracts frames from the video in full color.
3. Reduces resolution using OpenCV.
4. Saves extracted frames at a specified interval.

Requirements:
- OpenCV
- Python 3.x
"""

import cv2
import os
from tkinter import Tk, filedialog

class ColorFrameExtractor:
    def _init_(self, video_path, output_folder, interval_ms=100, resolution=(640, 480)):
        """
        Initialize the Color Frame Extractor.

        :param video_path: Path to the input video file.
        :param output_folder: Folder to save the extracted frames.
        :param interval_ms: Interval in milliseconds to extract frames.
        :param resolution: Tuple (width, height) to resize frames.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.interval_ms = interval_ms
        self.resolution = resolution

        # Create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        """
        Extract color frames from the video at the specified interval.
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * (self.interval_ms / 1000.0))

        frame_number = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Reduce resolution
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)

            if frame_number % frame_interval == 0:
                output_path = os.path.join(self.output_folder, f"frame_{saved_frame_count:04d}.png")
                cv2.imwrite(output_path, frame)
                saved_frame_count += 1

            frame_number += 1

        cap.release()
        print(f"Extracted {saved_frame_count} color frames to {self.output_folder}")

# Function to upload video file
def upload_video():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", ".mp4;.avi;.mov;.mkv")])
    return file_path

# Upload and extract frames
video_path = upload_video()
if video_path:
    extractor = ColorFrameExtractor(video_path, "frames_color", interval_ms=100, resolution=(640, 480))
    extractor.extract_frames()
# -- coding: utf-8 --
"""
Color Video Frame Extractor

Features:
1. Uploads a video file.
2. Extracts frames from the video in full color.
3. Reduces resolution using OpenCV.
4. Saves extracted frames at a specified interval.

Requirements:
- OpenCV
- Python 3.x
"""

import cv2
import os
from tkinter import Tk, filedialog

class ColorFrameExtractor:
    def _init_(self, video_path, output_folder, interval_ms=100, resolution=(640, 480)):
        """
        Initialize the Color Frame Extractor.

        :param video_path: Path to the input video file.
        :param output_folder: Folder to save the extracted frames.
        :param interval_ms: Interval in milliseconds to extract frames.
        :param resolution: Tuple (width, height) to resize frames.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.interval_ms = interval_ms
        self.resolution = resolution

        # Create output folder if not exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        """
        Extract color frames from the video at the specified interval.
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * (self.interval_ms / 1000.0))

        frame_number = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Reduce resolution
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)

            if frame_number % frame_interval == 0:
                output_path = os.path.join(self.output_folder, f"frame_{saved_frame_count:04d}.png")
                cv2.imwrite(output_path, frame)
                saved_frame_count += 1

            frame_number += 1

        cap.release()
        print(f"Extracted {saved_frame_count} color frames to {self.output_folder}")

# Function to upload video file
def upload_video():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", ".mp4;.avi;.mov;.mkv")])
    return file_path

# Upload and extract frames
video_path = upload_video()
if video_path:
    extractor = ColorFrameExtractor(video_path, "frames_color", interval_ms=100, resolution=(640, 480))
    extractor.extract_frames()