import cv2
import pytesseract
from PIL import Image
import re
import csv
import threading

# --- Functions ---

def extract_numbers_from_text(text):
    """Extract all numbers from the given text and return as a list."""
    return re.findall(r'\d+', text)

def extract_numbers_from_frame_with_timeout(cap, frame_count, timeout=5.0):
    """Extract numbers from a specific frame with a timeout."""
    result = []
    exception = None

    def worker():
        nonlocal result, exception
        try:
            # Read the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_count}")

            text = pytesseract.image_to_string(Image.fromarray(frame), lang='eng')
            result = extract_numbers_from_text(text)
        except Exception as e:
            exception = e

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f"Warning: OCR took longer than {timeout} seconds for frame {frame_count}. Skipping...")
        thread.join()  # Ensure the thread finishes before moving on
    if exception:
        print(f"Error occurred while processing frame {frame_count}: {exception}")

    return result

# --- Main Code ---

video_path = "./data/move.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Extract numbers every 5 seconds and store them along with the time in a list
data = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_skip = fps * 5

for frame_count in range(0, total_frames, frame_skip):
    time = frame_count / fps
    numbers = extract_numbers_from_frame_with_timeout(cap, frame_count)
    data.append((time, numbers))

cap.release()

# Save the extracted data to CSV
csv_filename = "./table/extracted_numbers.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time', 'Numbers'])
    for entry in data:
        csvwriter.writerow([entry[0], ' '.join(entry[1])])

