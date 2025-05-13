# Copyright (c) 2025 Mitchell Brenner
# Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
# See LICENSE for details.

import cv2
from collections import Counter
import numpy as np
from paddleocr import PaddleOCR

def blur_watermark(input_video_path, output_video_path):
    """
    Processes a video to detect and blur username watermarks in each frame.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path where the processed video will be saved.

    Returns:
        bool: True if a watermark was detected and blurred (more than 5 hits), False otherwise.

    Raises:
        IOError: If the input video cannot be opened.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # run OCR once every second
    interval = max(1, int(round(fps / 4)))
    frame_index = 0
    current_coords = None
    hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # only run detection on the first frame and then every `interval` frames
        if frame_index % interval == 0:
            found, coords = detect_username_with_paddle(frame)
            if found:
                current_coords = coords
                hits += 1

        # if we have a valid box, blur it
        if current_coords:
            x, y, w, h = current_coords

            # clamp to frame bounds
            x  = max(0, x)
            y  = max(0, y)
            w  = min(w, width  - x) + 10
            h  = min(h, height - y) + 5

            if w > 0 and h > 0:
                roi = frame[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(roi, (0, 0), 12)
                frame[y:y+h, x:x+w] = blurred

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print("Watermark blurred and saved to", output_video_path)

    if hits > 5:
        return True
    else:
        print("No watermark detected; skipping blur.")
        return False


def detect_username_with_paddle(frame):
    """
    Detects a username (preceded by '@') in a video frame using PaddleOCR.

    Args:
        frame (numpy.ndarray): The video frame in BGR color space.

    Returns:
        tuple:
            - bool: True if a username watermark was found, False otherwise.
            - tuple or None: Bounding box for the detected username in (x, y, w, h) or None.
    """

    # TODO: initialize PaddleOCR once and pass it to the function ( version issues right now)
    OCR = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False) 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Uncomment to see the image
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    result = OCR.ocr(img, cls=True)

    if not result or len(result) == 0:
        print("No text detected.")
        return False, None
    

    for line in result:

        if not line:
            continue

        for word_info in line:
            if word_info[1][0].startswith("@") and len(word_info[1][0]) > 1:
                print("Found Username:", word_info[1][0])
                
                box = np.array(word_info[0])
                # get the coordinates & width and height (x, y, w, h)
                x, y = int(box[0][0]), int(box[0][1])
                w, h = int(box[2][0] - box[0][0]), int(box[2][1] - box[0][1])

                return True, (x, y, w, h)
            
    print("No username detected.")
    return False, None


# Example usage to test the function
if __name__ == "__main__":
    INPUT = "data/raw/tiktok1.mp4"
    OUTPUT = "data/processed/blur_opencv.mp4"

    success = blur_watermark(INPUT, OUTPUT)
    print("Success:", success)