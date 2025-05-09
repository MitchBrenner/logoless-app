

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
        # Draw ROI again
        # cv2.rectangle(matched_frame,
        #               (roi_x, roi_y),
        #               (roi_x + roi_w, roi_y + roi_h),
        #               (255, 0, 0), 2)
        # # Draw watermark bounding box (green)
        # bottom_right = (matched_loc[0] + w, matched_loc[1] + h)
        # cv2.rectangle(matched_frame,
        #               matched_loc,
        #               bottom_right,
        #               (0, 255, 0), 3)
        # cv2.imshow('ROI and Detected Watermark', matched_frame)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True, (matched_loc[0], matched_loc[1], w, h)
    else:
        print('No watermark detected in the specified ROI.')
        return False, None



def blur_watermark_with_opencv(
    input_video_path: str,
    output_video_path: str,
    template_path: str,
    method=cv2.TM_CCORR,
    threshold=0.9,
    max_frames=1000,
    roi_x=0,
    roi_y=425,
    roi_w=None,
    roi_h=None,
    min_hits=300,
    blur_ksize=(0, 0),
    blur_sigma=15
) -> bool:
    """
    Detects watermark region via detect_watermark_in_roi, then reads each frame,
    applies a Gaussian blur to that region, and writes out a new video.
    Returns True if blurred, False otherwise.
    """
    # 1) Detect the watermark region
    found, coords = detect_watermark_in_roi(
        input_video_path,
        template_path,
        method=method,
        threshold=threshold,
        max_frames=max_frames,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
        min_hits=min_hits
    )
    if not found:
        print("No watermark detected; skipping blur.")
        return False

    x, y, w, h = coords

    # 2) Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    # Get video properties
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 3) Prepare output writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 4) Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract ROI and blur it
        roi = frame[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, blur_ksize, blur_sigma)
        frame[y:y+h, x:x+w] = blurred

        out.write(frame)

    # 5) Release resources
    cap.release()
    out.release()

    print(f"Blur applied at region {(x, y, w, h)}. Output: {output_video_path}")
    return True


if __name__ == "__main__":
    INPUT    = "data/raw/tiktok1.mp4"
    OUTPUT   = "data/processed/blur_opencv.mp4"
    TEMPLATE = "assets/templates/tiktok_watermark_cropped.png"

    success = blur_watermark_with_opencv(INPUT, OUTPUT, TEMPLATE)
    print("Success:", success)
