# Copyright (c) 2025 Mitchell Brenner
# Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
# See LICENSE for details.

import cv2
from collections import Counter
import easyocr
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt

"""
    Detects the location of a watermark in the specified region of a video
    using OpenCV template matching.

    This function scans multiple frames to confirm the presence of the
    watermark with a consistent confidence level, reducing false positives
    from random matches.

    Args:
        video_path (str): Path to the input video file.
        template_path (str): Path to the cropped watermark image template.
        method (int): OpenCV matching method (e.g. cv2.TM_CCORR).
        threshold (float): Match confidence threshold.
        max_frames (int): Max number of frames to scan.
        roi_x (int): X coordinate of the top-left corner of the ROI.
        roi_y (int): Y coordinate of the top-left corner of the ROI.
        roi_w (int or None): Width of the ROI. Defaults to template width + 5.
        roi_h (int or None): Height of the ROI. Defaults to template height + 5.
        min_hits (int): Number of confident matches required to confirm detection.

    Returns:
        Tuple[bool, Tuple[int, int, int, int] or None]:
            - True and bounding box (x, y, w, h) if watermark is detected
            - False and None if no watermark is founde
"""
def detect_watermark_in_roi(
    video_path,
    template_path,
    method=cv2.TM_CCORR,
    threshold=0.9,
    max_frames=360,
    roi_x=0,
    roi_y=0,
    roi_w=None,
    roi_h=None,
    min_hits=50
):
    # Load template in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    w, h = template.shape
    positions = []

    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set ROI 
    roi_w = w + 5
    roi_h = (height - h)

    frame_index = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame_index >= max_frames:
            break

        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]


        res = cv2.matchTemplate(roi, template, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            # Record the absolute location of each match
            absolute_loc = (roi_x + max_loc[0], roi_y + max_loc[1])
            positions.append(absolute_loc)

            # IF you want to see the matches in the video, uncomment the following lines
            # cv2.rectangle(
            #     frame,
            #     (absolute_loc[0], absolute_loc[1]),
            #     (absolute_loc[0] + w, absolute_loc[1] + h),
            #     (0, 255, 0),
            #     2
            # )
        # cv2.imshow('Video', frame)
        # cv2.waitKey(1)

    video.release()

    # Determine if most frequent match hits the threshold
    if positions:
        most_common_loc, count = Counter(positions).most_common(1)[0]
        if count >= min_hits:
            x, y = most_common_loc
            return True, (x, y, w, h)
    print('No watermark detected in the specified ROI.')
    return False, None

"""
    Applies a Gaussian blur to the detected watermark region in a video.
    This function uses OpenCV to read the video, apply the blur, and write
    the output to a new video file.
    Args:

        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output video file.
        template_path (str): Path to the cropped watermark image template.
        blur_ksize (tuple): Kernel size for Gaussian blur.
        blur_sigma (float): Standard deviation for Gaussian blur.
    Returns:
        bool: True if the watermark was blurred, False otherwise.
"""
def blur_watermark_with_opencv(
    input_video_path: str,
    output_video_path: str,
    template_path: str,
    blur_ksize=(0, 0),
    blur_sigma=5
) -> bool:

    found, coords1, coords2, = detect_username(input_video_path)
    if not found:
        print("No watermark detected; skipping blur.")
        return False

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    x1, y1, w1, h1 = coords1
    if coords2 is None:
        # TODO: create a better fallback incase the second username is not found
        # x2, y2, w2, h2 = width - x1, y1 + 200, w1, h1
        x2, y2, w2, h2 = x1, y1, w1, h1 
    else: 
        x2, y2, w2, h2 = coords2


    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")


    frame_index = 0
    blur_switch_frame = int(fps * 5) + 1  # 5 seconds in

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # choose which box to blur
        if frame_index < blur_switch_frame:
            cx, cy, cw, ch = x1, y1, w1, h1
        else:
            cx, cy, cw, ch = x2, y2, w2, h2

            # Uncomment to see the rectangle drawn on the video 
            # cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)

        # apply blur to that region
        roi = frame[cy:cy + ch, cx:cx + cw]
        blurred = cv2.GaussianBlur(roi, blur_ksize, blur_sigma)
        frame[cy:cy + ch, cx:cx + cw] = blurred

        out.write(frame)
        frame_index += 1

    out.release()

    return True

"""
    Detects the username in a video using Tesseract OCR.
    This function scans the first 5 seconds of the video and then
    seeks to the 5-second mark to find the username.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        Tuple[bool, Tuple[int, int, int, int] or None, Tuple[int, int, int, int] or None]:
            - True and bounding boxes (x, y, w, h) for both usernames if detected
            - False and None if no username is found
"""
def detect_username(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    first_coords = None

    # 1) Scan first 5 seconds for first username
    for _ in range(int(fps * 5)):
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT, config='--psm 6'
        )
        for i, txt in enumerate(data['text']):
            if txt.strip() == "@" and i + 1 < len(data['text']):
                x0, y0 = data['left'][i], data['top'][i]
                x1, y1 = data['left'][i+1], data['top'][i+1]
                w1, h1 = data['width'][i+1], data['height'][i+1]
                # combined box from '@' to username end
                x = x0
                y = min(y0, y1)
                w = (x1 + w1) - x0
                h = max(y0 + data['height'][i], y1 + h1) - y
                first_coords = (x, y, w, h)
                break
        if first_coords:
            break

    # seek to 5-second mark for second username
    video.set(cv2.CAP_PROP_POS_MSEC, 5080)
    second_coords = None
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT, config='--psm 6'
        )
        for i, txt in enumerate(data['text']):
            if txt.strip() == "@" and i + 1 < len(data['text']):
                x0, y0 = data['left'][i], data['top'][i]
                x1, y1 = data['left'][i+1], data['top'][i+1]
                w1, h1 = data['width'][i+1], data['height'][i+1]
                x = x0
                y = min(y0, y1)
                w = (x1 + w1) - x0
                h = max(y0 + data['height'][i], y1 + h1) - y
                second_coords = (x, y, w, h)
                break
        if second_coords:
            break

    video.release()

    if not first_coords and not second_coords:
        print("No username detected.")
        return False, None, None
    print(f"Username detected at {first_coords} and {second_coords}.")
    return True, first_coords, second_coords


def blur_watermark(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

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
            w  = min(w, width  - x) + 5
            h  = min(h, height - y) + 5

            if w > 0 and h > 0:
                roi     = frame[y:y+h, x:x+w]
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
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # image_path = "data/raw/reduced_quality_img.jpg"


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Uncomment to see the image
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    result = ocr.ocr(img, cls=True)

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
                print("Box:", box)
                # get the coordinates & width and height (x, y, w, h)
                x, y = int(box[0][0]), int(box[0][1])
                w, h = int(box[2][0] - box[0][0]), int(box[2][1] - box[0][1])
                print(x,y, w, h)
                # text = word_info[1][0]
                # confidence = word_info[1][1]
                # draw the box
                # pts = np.array(box, dtype=np.int32)    # or dtype=np.intp
                # cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                return True, (x, y, w, h)
            
    print("No username detected.")
    return False, None

    # plt.figure()
    # plt.imshow(img)
    # plt.show()

# Example usage to test the function
if __name__ == "__main__":
    INPUT = "data/raw/no_watermark.mp4"
    OUTPUT = "data/processed/blur_opencv.mp4"
    # TEMPLATE = "assets/templates/tiktok_watermark_cropped.png"

    # success = blur_watermark_with_opencv(INPUT, OUTPUT, TEMPLATE)
    # print("Success:", success)
    # detect_username("data/raw/tiktok3.mp4")
    # detect_username("data/raw/tiktok1.mp4")
    # detect_username_with_paddle()

    success = blur_watermark(INPUT, OUTPUT)
    print("Success:", success)