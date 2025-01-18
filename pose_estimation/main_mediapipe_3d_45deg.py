"""
Pose estimation with MediaPipe Pose, outputting 3D coordinates (x, y, z).
Displays:
 - The normal pose overlay
 - An additional view where the skeleton is rotated 45° in the horizontal plane.

:author: Your Name
:date: 2025-01-18
"""

import cv2
import time
import numpy as np
import mediapipe as mp

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(rgb_frame)

    if not results.pose_landmarks:
        return np.array([])

    landmarks = results.pose_landmarks.landmark
    output = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        output[i, 0] = lm.x
        output[i, 1] = lm.y
        output[i, 2] = lm.z
        output[i, 3] = lm.visibility
    return output


def rotate_landmarks_y_axis(
    landmarks_3d: np.ndarray, angle_degrees: float
) -> np.ndarray:
    """
    Rotate the 3D landmarks around the Y axis by a given angle (in degrees).
    MediaPipe convention:
       x: [0..1] left->right
       y: [0..1] top->bottom
       z: negative is "toward the camera" (roughly)
    Rotating about Y means a "horizontal-plane" rotation.

    :param landmarks_3d: (33, 4) array [x, y, z, visibility].
    :param angle_degrees: rotation angle around Y axis in degrees.
    :return: A (33, 4) array of rotated 3D landmarks (x', y', z', visibility).
    """
    theta = np.radians(angle_degrees)
    # Rotation matrix around Y:
    #  [ cosθ   0   sinθ ]
    #  [   0    1     0  ]
    #  [-sinθ   0   cosθ ]
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=np.float32,
    )

    rotated = np.zeros_like(landmarks_3d)
    for i in range(33):
        x, y, z, vis = landmarks_3d[i]
        # Apply rotation to (x, y, z) only
        xyz_in = np.array([x, y, z], dtype=np.float32)
        xyz_out = R @ xyz_in
        rotated[i, :3] = xyz_out
        rotated[i, 3] = vis
    return rotated


def draw_pose_2d_custom(
    image: np.ndarray,
    landmarks_3d: np.ndarray,
    connections,
    visibility_threshold=0.5,
    color=(0, 255, 0),
):
    """
    Draw 2D points + connections onto an image using the landmarks' (x,y).
    Assumes x,y in [0..1].

    :param image: The BGR image to draw onto.
    :param landmarks_3d: (33, 4) array -> [x, y, z, visibility].
    :param connections: A list (or set) of landmark index pairs to connect.
    :param visibility_threshold: Only draw if both endpoints exceed this visibility.
    :param color: BGR color for lines/circles.
    """
    h, w, _ = image.shape
    # Draw connections first
    for i1, i2 in connections:
        vis1 = landmarks_3d[i1, 3]
        vis2 = landmarks_3d[i2, 3]
        if vis1 < visibility_threshold or vis2 < visibility_threshold:
            continue
        x1 = int(landmarks_3d[i1, 0] * w)
        y1 = int(landmarks_3d[i1, 1] * h)
        x2 = int(landmarks_3d[i2, 0] * w)
        y2 = int(landmarks_3d[i2, 1] * h)
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # Draw landmark points
    for i in range(len(landmarks_3d)):
        x, y, z, visibility = landmarks_3d[i]
        if visibility < visibility_threshold:
            continue
        px = int(x * w)
        py = int(y * h)
        cv2.circle(image, (px, py), 5, color, -1)


def pose_estimation_3d_demo(
    video_path: str, scale_factor: float = 1.0, show_mediapipe_overlay: bool = True
) -> None:
    """
    Demonstrates 3D pose estimation using MediaPipe Pose on a video.
    Also shows a second window with the pose rotated by 45° around Y.

    :param video_path: Path to the input video file.
    :param scale_factor: Factor by which to scale the frame for display.
    :param show_mediapipe_overlay: If True, use MediaPipe's own drawing function in the main view.
    :return: None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        time.sleep(0.03)  # for demonstration
        ret, frame = cap.read()
        if not ret:
            # Restart video for demonstration, or break:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                break

        # Optionally resize for display
        if scale_factor != 1.0:
            disp_width = int(frame.shape[1] * scale_factor)
            disp_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(
                frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA
            )

        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # The main display
        display_frame = frame.copy()

        # If a person is detected, handle 3D landmarks
        if results.pose_landmarks:
            # Convert landmarks to numpy
            landmarks_3d = np.array(
                [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )

            # Show normal pose overlay in the main frame
            if show_mediapipe_overlay:
                disp_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    disp_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                display_frame = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
            else:
                draw_pose_2d_custom(
                    display_frame,
                    landmarks_3d,
                    mp_pose.POSE_CONNECTIONS,
                    visibility_threshold=0.5,
                    color=(0, 255, 0),
                )

            # ---------------
            # 1) Rotate the 3D landmarks around Y by +45 degrees
            # 2) Project them onto a new blank image
            # ---------------
            angle_deg = -20.0
            rotated_3d = rotate_landmarks_y_axis(landmarks_3d, angle_deg)

            # Make a blank image for showing the rotated skeleton
            # You can choose a different size or background if you like
            h, w, _ = display_frame.shape
            rotated_view = np.zeros((h, w, 3), dtype=np.uint8)

            # IMPORTANT: the rotated x,y,z are still in 'normalized' space.
            # If the rotation pushes x,y out of [0..1], we won't see them.
            # For a basic demo, let's simply clamp them to [0..1].
            # For a more advanced approach, you might recenter or scale them.
            rotated_3d_clamped = rotated_3d.copy()
            rotated_3d_clamped[:, 0] = np.clip(rotated_3d_clamped[:, 0], 0, 1)
            rotated_3d_clamped[:, 1] = np.clip(rotated_3d_clamped[:, 1], 0, 1)
            # z is not used in direct 2D drawing, but you might also clamp or scale it.

            # Now draw on that blank image using the same connections
            draw_pose_2d_custom(
                rotated_view,
                rotated_3d_clamped,
                mp_pose.POSE_CONNECTIONS,
                visibility_threshold=0.5,
                color=(0, 255, 255),
            )

            # Show that second image in a new window
            cv2.imshow("Rotated Pose (45 deg)", rotated_view)

        # Show the main view
        cv2.imshow("3D Pose Estimation (MediaPipe)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


def main():
    """
    Main function that runs the 3D pose estimation on a sample video.
    Displays two windows:
      - Normal pose overlay
      - Additional view with a 45° rotation
    """
    video_path = "../data/deadlift_diagonal_view.mp4"
    pose_estimation_3d_demo(
        video_path=video_path, scale_factor=1.0, show_mediapipe_overlay=True
    )


if __name__ == "__main__":
    main()
