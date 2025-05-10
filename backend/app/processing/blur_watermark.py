import cv2
from collections import Counter


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


def blur_watermark_with_opencv(
    input_video_path: str,
    output_video_path: str,
    template_path: str,
    blur_ksize=(0, 0),
    blur_sigma=12
) -> bool:
    found, coords = detect_watermark_in_roi(
        input_video_path,
        template_path,
    )
    if not found:
        print("No watermark detected; skipping blur.")
        return False

    x, y, w, h = coords

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0
    blur_switch_frame = int(fps * 5) + 1  # 5 seconds in

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Default location (initial watermark position)
        current_x, current_y = x, y

        # After 5 seconds, switch to new position
        if frame_index >= blur_switch_frame:
            current_x = width - w + 20 
            current_y = height - h - 120 

        roi = frame[current_y:current_y + h, current_x:current_x + w]
        blurred = cv2.GaussianBlur(roi, blur_ksize, blur_sigma)
        frame[current_y:current_y + h, current_x:current_x + w] = blurred

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    print(f"Blur applied at {(x, y, w, h)} initially, then moved after 5s.")
    return True



# Example usage to test the function
if __name__ == "__main__":
    INPUT = "data/raw/tiktok1.mp4"
    OUTPUT = "data/processed/blur_opencv.mp4"
    TEMPLATE = "assets/templates/tiktok_watermark_cropped.png"

    success = blur_watermark_with_opencv(INPUT, OUTPUT, TEMPLATE)
    print("Success:", success)