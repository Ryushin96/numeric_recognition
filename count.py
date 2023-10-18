import cv2
import pytesseract
from PIL import Image
import re
import csv
import threading
import queue
# --- Environment variables ---
INPUT_PATH = "./data/video.mov"
OUTPUT_PATH_PREFIX = "cropped_video_"
CROP_RECTS = [
    (2630, 1522, 100, 38),
    (2150, 1640, 100, 38),
    (2150, 1592, 100, 38),
    (2630, 1592, 100, 38),
    (2630, 1635, 100, 38)
]
TARGET_FPS = 0.2 
SKIP_SECONDS = 5  # 10秒ごとに1フレームを読み取る
CSV_FILENAME_PREFIX = "./table/extracted_numbers_"

# --- Functions ---

def extract_numbers_from_text(text):
    """Extract all numbers from the given text and return as a list."""
    return re.findall(r'\d+', text)

def extract_numbers_from_frame(frame, result_queue, idx):
    """Extract numbers from a frame and put the result in a queue."""
    text = pytesseract.image_to_string(Image.fromarray(frame), lang='eng')
    numbers = extract_numbers_from_text(text)
    result_queue.put((idx, numbers))

def extract_and_save_numbers_for_rect(input_path):
    """Extract numbers from the frames of a video for specific rectangles."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_data = {idx: [] for idx in range(len(CROP_RECTS))}
    frame_skip = original_fps * SKIP_SECONDS

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            # Display progress
            progress_percentage = (frame_number / total_frames) * 100
            print(f"Processing frame {frame_number} of {total_frames} ({progress_percentage:.2f}% completed)")

            time = frame_number / original_fps
            result_queue = queue.Queue()
            threads = []

            for idx, crop_rect in enumerate(CROP_RECTS):
                cropped_frame = frame[crop_rect[1]:crop_rect[1]+crop_rect[3], crop_rect[0]:crop_rect[0]+crop_rect[2]]
                thread = threading.Thread(target=extract_numbers_from_frame, args=(cropped_frame, result_queue, idx))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            while not result_queue.empty():
                idx, numbers = result_queue.get()
                all_data[idx].append((time, numbers))

        frame_number += 1

    cap.release()

    for idx, data in all_data.items():
        csv_filename = CSV_FILENAME_PREFIX + str(idx + 1) + ".csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Time', 'Numbers'])
            for entry in data:
                csvwriter.writerow([entry[0], ' '.join(entry[1])])

# --- Main Code ---

extract_and_save_numbers_for_rect(INPUT_PATH)

