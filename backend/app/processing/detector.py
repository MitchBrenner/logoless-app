# Copyright (c) 2025 Mitchell Brenner
# Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
# See LICENSE for details.

import cv2
 
def detect_watermark_in_roi(
    video_path,
    template_path,
    method=cv2.TM_CCORR,
    threshold=0.9,
    max_frames=1000,
    roi_x=0,
    roi_y=425,
    roi_w=None,
    roi_h=None,
    min_hits=300
):
    # Load template in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    w, h = template.shape

    # Set ROI width and height if not provided
    if roi_w is None:
        roi_w = w + 5
    if roi_h is None:
        roi_h = h + 5

    # Open video capture
    video = cv2.VideoCapture(video_path)
    hits = 0
    matched_frame = None
    matched_loc = None

    frame_index = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame_index >= max_frames:
            break

        frame_index += 1

        # Draw ROI on the frame for visual reference
        cv2.rectangle(frame,
                      (roi_x, roi_y),
                      (roi_x + roi_w, roi_y + roi_h),
                      (255, 0, 0), 2)

        # Convert to grayscale and crop to ROI
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # Perform template matching in ROI
        res = cv2.matchTemplate(roi, template, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Increment hit counter if above threshold
        if max_val >= threshold:
            hits += 1
            if hits >= min_hits:
                # Compute absolute top-left coordinates
                matched_loc = (roi_x + max_loc[0], roi_y + max_loc[1])
                matched_frame = frame.copy()
                break

    # Release video capture
    video.release()

    # If a match was found, draw and show the result
    if matched_frame is not None:
        cv2.destroyAllWindows()
        return True, (matched_loc[0], matched_loc[1], w, h)
    else:
        print('No watermark detected in the specified ROI.')
        return False, None



# Example usage
if __name__ == "__main__":
    video_path = 'data/raw/no_tiktok_watermark.mp4'
    template_path = 'assets/templates/tiktok_watermark_cropped.png'
    print(detect_watermark_in_roi(video_path, template_path))
