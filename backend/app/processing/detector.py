import cv2

 
# All the 6 methods for comparison in a list
# methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
            # 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

# Configuration
VIDEO_PATH = 'data/raw/no_tiktok_watermark.mp4'
# VIDEO_PATH = 'data/raw/tiktok4.mp4'
TEMPLATE_PATH = 'assets/templates/tiktok_watermark_cropped.png'
METHOD = cv2.TM_CCORR # cv2.TM_CCOEFF_NORMED
THRESHOLD = .9
MAX_FRAMES = 1000


# Load template in grayscale
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
w, h = template.shape

# Open video capture
video = cv2.VideoCapture(VIDEO_PATH)
matched_frame = None
matched_loc = None

# Define the only region of interest (ROI)
ROI_X, ROI_Y = 0, 425   # top-left corner of ROI
ROI_W, ROI_H = w + 5, h + 5 # width & height of ROI

frame_index = 0
hits = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret or frame_index >= MAX_FRAMES:
        break

    frame_index += 1

    # Draw ROI on the frame for visual reference
    cv2.rectangle(frame,
                  (ROI_X, ROI_Y),
                  (ROI_X + ROI_W, ROI_Y + ROI_H),
                  (255, 0, 0), 2)

    # Convert to grayscale and crop to ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi  = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

    # Perform template matching inside the ROI
    res = cv2.matchTemplate(roi, template, METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    x,y = max_loc
    # Check if the match exceeds the threshold
    if max_val >= THRESHOLD :
        hits += 1
        if hits >= 300:
        # Compute absolute top-left coordinates on full frame
            top_left = (ROI_X + max_loc[0], ROI_Y + max_loc[1])
            matched_frame = frame.copy()
            matched_loc = top_left
            break

# Release video capture
video.release()

# If a match was found, draw and show the result
if matched_frame is not None:
    # Draw ROI again
    cv2.rectangle(matched_frame,
                  (ROI_X, ROI_Y),
                  (ROI_X + ROI_W, ROI_Y + ROI_H),
                  (255, 0, 0), 2)
    # Draw watermark bounding box (green)
    bottom_right = (matched_loc[0] + w, matched_loc[1] + h)
    cv2.rectangle(matched_frame,
                  matched_loc,
                  bottom_right,
                  (0, 255, 0), 3)
    cv2.imshow('ROI and Detected Watermark', matched_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('No watermark detected in the specified ROI.')
