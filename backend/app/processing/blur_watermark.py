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
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    w, h = template.shape

    if roi_w is None:
        roi_w = w + 5
    if roi_h is None:
        roi_h = h + 5

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        res = cv2.matchTemplate(roi, template, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            hits += 1
            if hits >= min_hits:
                matched_loc = (roi_x + max_loc[0], roi_y + max_loc[1])
                matched_frame = frame.copy()
                break

    video.release()

    if matched_frame is not None:
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

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0
    blur_switch_frame = int(fps * 5)  # 5 seconds in

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Default location (initial watermark position)
        current_x, current_y = x, y

        # After 5 seconds, switch to new position
        if frame_index >= blur_switch_frame:
            current_x = width - w + 20  # move more to the right
            current_y = height - h - 120  # move slightly higher

        roi = frame[current_y:current_y + h, current_x:current_x + w]
        blurred = cv2.GaussianBlur(roi, blur_ksize, blur_sigma)
        frame[current_y:current_y + h, current_x:current_x + w] = blurred

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    print(f"Blur applied at {(x, y, w, h)} initially, then moved after 5s.")
    return True


if __name__ == "__main__":
    INPUT = "data/raw/tiktok1.mp4"
    OUTPUT = "data/processed/blur_opencv.mp4"
    TEMPLATE = "assets/templates/tiktok_watermark_cropped.png"

    success = blur_watermark_with_opencv(INPUT, OUTPUT, TEMPLATE)
    print("Success:", success)