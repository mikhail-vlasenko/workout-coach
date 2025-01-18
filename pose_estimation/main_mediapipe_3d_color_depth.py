"""
Pose estimation with MediaPipe Pose, outputting 3D coordinates (x, y, z).

:author: Your Name
:date: 2025-01-18
"""

import cv2
import time
import numpy as np
import mediapipe as mp

# MediaPipe Pose includes 33 keypoints (landmarks).
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def detect_keypoints_mediapipe_3d(frame: np.ndarray) -> np.ndarray:
    """
    Detect 3D pose landmarks using MediaPipe Pose.

    :param frame: BGR input frame (e.g., from OpenCV).
    :return: A NumPy array of shape (33, 4), where each row is [x, y, z, visibility].
             - x, y, z are normalized [0..1], with z being a relative depth (negative = in front).
             - visibility is a float [0..1] indicating landmark confidence.
             If no person is detected, returns an empty array.
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run MediaPipe Pose detection
    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(rgb_frame)

    if not results.pose_landmarks:
        # Nothing detected
        return np.array([])

    # Extract 33 landmarks: each has x, y, z, visibility
    landmarks = results.pose_landmarks.landmark
    output = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        output[i, 0] = lm.x
        output[i, 1] = lm.y
        output[i, 2] = lm.z
        output[i, 3] = lm.visibility

    return output


def z_to_color(z: float) -> tuple[int, int, int]:
    """
    Convert a normalized z-value (often in range ~[-0.5..0.5], but can vary)
    into a color (B, G, R) for visualization.

    Negative z (closer to camera) will be more reddish; positive z (further away)
    will be more bluish in this example.

    :param z: The normalized depth value from MediaPipe Pose (z < 0 is in front).
    :return: A color in (B, G, R) format.
    """
    # Clamp z to a reasonable range for display (e.g., [-0.5, 0.5])
    z_clamped = max(-0.5, min(0.5, z))
    # Map z from [-0.5, 0.5] to [0, 1]
    normalized = (z_clamped + 0.5) / 1.0

    # We'll create a simple gradient:
    #   z = -0.5 => (B, G, R) = (0, 0, 255)   (red)
    #   z =  0.5 => (B, G, R) = (255, 0, 0)   (blue)
    # Feel free to adjust for your preferred color mapping.
    b_val = int(normalized * 255)  # goes from 0 to 255
    r_val = 255 - b_val  # goes from 255 down to 0
    g_val = 0

    return (b_val, g_val, r_val)


def draw_keypoints_2d(
    frame: np.ndarray, landmarks_3d: np.ndarray, confidence_threshold: float = 0.5
) -> None:
    """
    Draw 2D landmarks on the image (using only x,y) and color-code them by z-depth.

    :param frame: The image (BGR) on which to draw.
    :param landmarks_3d: A NumPy array of shape (33, 4) -> [x, y, z, visibility].
    :param confidence_threshold: Minimum visibility to draw the point.
    :return: None (draws in-place).
    """
    h, w, _ = frame.shape

    for i in range(len(landmarks_3d)):
        x, y, z, visibility = landmarks_3d[i]
        if visibility < confidence_threshold:
            continue

        px = int(x * w)
        py = int(y * h)

        # Convert z to a color
        color = z_to_color(z)
        cv2.circle(frame, (px, py), 5, color, -1)


def pose_estimation_3d_demo(
    video_path: str, scale_factor: float = 1.0, show_mediapipe_overlay: bool = True
) -> None:
    """
    Demonstrates 3D pose estimation using MediaPipe Pose on a video.

    :param video_path: Path to the input video file.
    :param scale_factor: Factor by which to scale the frame for display.
    :param show_mediapipe_overlay: If True, use MediaPipe's own drawing function (no color-coding).
    :return: None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize MediaPipe Pose once outside the loop for performance.
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        time.sleep(0.03)  # for demonstration, limit frame rate slightly
        ret, frame = cap.read()
        if not ret:
            # End of video or read error, optionally restart or break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Optionally resize for display
        if scale_factor != 1.0:
            disp_width = int(frame.shape[1] * scale_factor)
            disp_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(
                frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA
            )

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # If a person is detected, extract keypoints
        if results.pose_landmarks:
            # Convert landmarks to numpy array
            landmarks_3d = np.array(
                [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )

            if show_mediapipe_overlay:
                # Use built-in MediaPipe drawing for convenience (no color-coding)
                disp_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    disp_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                frame = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Our custom color-coded drawing
                draw_keypoints_2d(frame, landmarks_3d, confidence_threshold=0.5)

        cv2.imshow("3D Pose Estimation (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


def main():
    """
    Main function that runs the 3D pose estimation on a sample video.
    """
    # Replace with your own video file or use your webcam with 0
    video_path = "../data/deadlift_diagonal_view.mp4"

    pose_estimation_3d_demo(
        video_path=video_path,
        scale_factor=1.0,
        # If True, draws MediaPipe's official overlay (no color-coding).
        # If False, uses our custom color-coded drawing by Z depth.
        show_mediapipe_overlay=False,
    )


if __name__ == "__main__":
    main()
