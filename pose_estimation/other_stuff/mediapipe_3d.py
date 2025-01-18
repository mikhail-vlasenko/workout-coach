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
# We'll import the drawing utils for convenience if you want built-in visualization.
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


def draw_keypoints_2d(
    frame: np.ndarray, landmarks_3d: np.ndarray, confidence_threshold: float = 0.5
) -> None:
    """
    Draw 2D landmarks on the image (using only x,y) and some edges.

    :param frame: The image (BGR) on which to draw.
    :param landmarks_3d: A NumPy array of shape (33, 4) -> [x, y, z, visibility].
    :param confidence_threshold: Minimum visibility to draw the point.
    :return: None (draws in-place).
    """
    h, w, _ = frame.shape

    # You could define your own skeleton edges if you like.
    # Here we will just do something simple: draw every point, no lines,
    # or you can rely on mp_drawing to do it for you.
    for i in range(len(landmarks_3d)):
        x, y, z, visibility = landmarks_3d[i]
        if visibility < confidence_threshold:
            continue
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)


def pose_estimation_3d_demo(
    video_path: str, scale_factor: float = 1.0, show_mediapipe_overlay: bool = True
) -> None:
    """
    Demonstrates 3D pose estimation using MediaPipe Pose on a video.

    :param video_path: Path to the input video file.
    :param scale_factor: Factor by which to scale the frame for display.
    :param show_mediapipe_overlay: If True, use MediaPipe's own drawing function.
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
        time.sleep(0.03)  # for demonstration
        ret, frame = cap.read()
        if not ret:
            # End of video or read error
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            # break

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Optionally resize for display
        if scale_factor != 1.0:
            disp_width = int(frame.shape[1] * scale_factor)
            disp_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(
                frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA
            )

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

            # Draw the keypoints (2D projection)
            if show_mediapipe_overlay:
                # Use built-in MediaPipe drawing for convenience
                # Must convert the display frame back to RGB for mp_drawing
                disp_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    disp_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                frame = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
            else:
                draw_keypoints_2d(frame, landmarks_3d, confidence_threshold=0.5)

            # You now have 3D info in `landmarks_3d`. For example, you can print the first keypoint's z:
            # print("Landmark 0 (nose) => x=%.3f, y=%.3f, z=%.3f, visibility=%.3f" %
            #       (landmarks_3d[0, 0], landmarks_3d[0, 1], landmarks_3d[0, 2], landmarks_3d[0, 3]))

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
    video_path = "../../data/deadlift_diagonal_view.mp4"

    pose_estimation_3d_demo(
        video_path=video_path, scale_factor=1.0, show_mediapipe_overlay=True
    )


if __name__ == "__main__":
    main()
