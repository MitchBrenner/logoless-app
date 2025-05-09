import cv2
from detector import detect_watermark_in_roi

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
    INPUT    = "data/raw/nowatermark.mp4"
    OUTPUT   = "data/processed/blur_opencv.mp4"
    TEMPLATE = "assets/templates/tiktok_watermark_cropped.png"

    success = blur_watermark_with_opencv(INPUT, OUTPUT, TEMPLATE)
    print("Success:", success)
