import cv2
import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub

MOVENET_MODEL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

EDGES = [
    (0, 1),
    (0, 2),  # nose to eyes
    (1, 3),
    (2, 4),  # eyes to ears
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left shoulder -> left elbow -> left wrist
    (6, 8),
    (8, 10),  # right shoulder -> right elbow -> right wrist
    (5, 11),
    (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left hip -> left knee -> left ankle
    (12, 14),
    (14, 16),  # right hip -> right knee -> right ankle
]


def draw_keypoints_and_edges(
    frame: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.2
) -> None:
    """
    Draws keypoints and skeletal connections on a frame.

    :param frame: The image (BGR) on which to draw.
    :param keypoints: Numpy array of shape (17, 3) containing normalized (y, x, confidence),
                      where y, x are in [0..1] with respect to the original frame dimensions.
    :param confidence_threshold: Minimum confidence for a keypoint to be drawn.
    :return: None. Draws directly on the `frame`.
    """
    height, width, _ = frame.shape

    # Draw keypoints
    for idx, (y, x, conf) in enumerate(keypoints):
        if conf < confidence_threshold:
            continue
        cx, cy = int(x * width), int(y * height)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Draw edges
    for p1, p2 in EDGES:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if (c1 >= confidence_threshold) and (c2 >= confidence_threshold):
            start_point = (int(x1 * width), int(y1 * height))
            end_point = (int(x2 * width), int(y2 * height))
            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)


def detect_keypoints_movenet(
    movenet_signature, frame: np.ndarray, input_size: int = 256
) -> np.ndarray:
    """
    Run inference on a single frame using MoveNet with letterboxing, then correct for
    the padding so keypoints accurately map to the original frame's dimensions.

    :param movenet_signature: The serving signature function from the loaded MoveNet TF Hub model.
    :param frame: BGR frame from OpenCV.
    :param input_size: The square size for MoveNet input (e.g., 256 for Thunder).
    :return: Numpy array of shape (17, 3) with (y, x, confidence) normalized to the
             original frame dimensions (y,x in [0..1]).
    """
    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = frame.shape[:2]

    # Letterboxed resize with pad to keep aspect ratio in a 256x256 region
    resized = tf.image.resize_with_pad(
        tf.expand_dims(image_rgb, axis=0), input_size, input_size
    )
    input_tensor = tf.cast(resized, dtype=tf.int32)

    # Run inference
    outputs = movenet_signature(input_tensor)
    # 'output_0' is [1, 1, 17, 3] -> (y, x, confidence)
    keypoints_letterboxed = outputs["output_0"].numpy()[0, 0, :, :]

    # Undo the letterboxing to get correct positions in the original frame
    # 1) Determine how we scaled the original frame to fit in 256x256
    scale = input_size / max(h_orig, w_orig)
    new_h = int(round(h_orig * scale))
    new_w = int(round(w_orig * scale))

    # 2) Determine the padding offset
    # if the frame was wide, black bars are top/bottom -> pad_y
    # if the frame was tall, black bars are left/right -> pad_x
    pad_y = (input_size - new_h) / 2.0
    pad_x = (input_size - new_w) / 2.0

    corrected_keypoints = np.zeros_like(keypoints_letterboxed)

    # Each keypoint_letterboxed is in normalized coords [0..1] for the 256x256 box
    # We'll map back to [0..1] in the original frame
    for i in range(17):
        y, x, conf = keypoints_letterboxed[i]

        # Convert from [0..1] in 256x256 to absolute [0..256] coords
        y_256 = y * input_size
        x_256 = x * input_size

        # Subtract offset
        y_no_pad = y_256 - pad_y
        x_no_pad = x_256 - pad_x

        # Scale back to the original resolution
        y_orig = y_no_pad / scale
        x_orig = x_no_pad / scale

        # Normalize to [0..1] with respect to original frame dims
        y_norm = y_orig / h_orig
        x_norm = x_orig / w_orig

        corrected_keypoints[i] = [y_norm, x_norm, conf]

    return corrected_keypoints


def pose_estimation_subsample(
    video_path: str, subsample_rate: int = 1, scale_factor: float = 1.0
) -> None:
    """
    Pose estimation on subsampled frames of a video using MoveNet.

    :param video_path: Path to the input video file.
    :param subsample_rate: Read every nth frame (default: 1).
    :param scale_factor: Factor by which to scale the frame for display (default: 1.0 = no scaling).
    :return: None. Displays frames with pose landmarks drawn on them.
    """
    # Load the MoveNet model
    movenet_model = hub.load(MOVENET_MODEL)
    movenet_signature = movenet_model.signatures["serving_default"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_idx = 0

    while True:
        time.sleep(0.1)  # Just for demonstration (avoid spamming)
        ret, frame = cap.read()
        if not ret:
            # End of video or read error; try to re-open or break
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            # break  # Uncomment this if you want to stop at the end

        # Only process every 'subsample_rate' frame
        if frame_idx % subsample_rate == 0:
            # Detect keypoints using MoveNet (letterboxing + correction)
            keypoints = detect_keypoints_movenet(movenet_signature, frame)

            # Draw keypoints/edges
            draw_keypoints_and_edges(frame, keypoints, confidence_threshold=0.2)

            # Scale the annotated frame if needed
            if scale_factor != 1.0:
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )

            cv2.imshow("MoveNet Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function to demonstrate the pose_estimation_subsample function with MoveNet.

    :return: None.
    """
    video_path = (
        "../data/deadlift_diagonal_view.mp4"  # Replace with your video file path
    )
    subsample_rate = 1  # Process every frame
    scale_factor = 2.0  # Scale for display if desired
    pose_estimation_subsample(video_path, subsample_rate, scale_factor)


if __name__ == "__main__":
    main()
