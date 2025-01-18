import cv2
import time
import numpy as np
import mediapipe as mp

# MediaPipe Pose includes 33 keypoints (landmarks).
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def normalize_pose_orientation(
    landmarks_3d: np.ndarray,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Normalize the pose by aligning the hips to the origin and the torso to a canonical axis.

    :param landmarks_3d: Numpy array of shape (33, 4) with (x, y, z, visibility).
    :param confidence_threshold: Minimum visibility to consider a landmark.
    :return: Numpy array of shape (33, 3) with normalized coordinates.
    """
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    left_hip = landmarks_3d[LEFT_HIP]
    right_hip = landmarks_3d[RIGHT_HIP]
    left_shoulder = landmarks_3d[LEFT_SHOULDER]
    right_shoulder = landmarks_3d[RIGHT_SHOULDER]

    if left_hip[3] < confidence_threshold or right_hip[3] < confidence_threshold:
        raise ValueError("Low confidence in hip landmarks, cannot normalize orientation.")

    # Compute torso origin (hip midpoint)
    hip_center = (left_hip[:3] + right_hip[:3]) / 2

    # Define torso vector (hips -> shoulders) as the Y-axis
    shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2
    torso_y = shoulder_center - hip_center
    torso_y /= np.linalg.norm(torso_y)

    # Define X-axis as perpendicular to the hip line
    hip_line = right_hip[:3] - left_hip[:3]
    torso_x = np.cross(hip_line, torso_y)
    torso_x /= np.linalg.norm(torso_x)

    # Define Z-axis as orthogonal to both X and Y (right-hand rule)
    torso_z = np.cross(torso_x, torso_y)
    torso_z /= np.linalg.norm(torso_z)

    # Rotation matrix to align torso with canonical axes
    rotation_matrix = np.vstack([torso_x, torso_y, torso_z]).T

    # Normalize all landmarks
    normalized_landmarks = np.zeros((landmarks_3d.shape[0], 3), dtype=np.float32)
    for i, (x, y, z, visibility) in enumerate(landmarks_3d):
        if visibility < confidence_threshold:
            continue

        position = np.array([x, y, z]) - hip_center  # Translate to origin
        position = rotation_matrix @ position        # Rotate to align torso
        normalized_landmarks[i] = position

    return normalized_landmarks

def draw_connections(landmarks, view, edges, color, scale=1.0):
    """
    Draw connections (lines) between landmarks in a 2D plane.

    :param landmarks: Numpy array of shape (33, 3).
    :param view: The 2D plane image to draw on.
    :param edges: List of landmark pairs (connections).
    :param color: Line color (BGR tuple).
    :param scale: Scaling factor for landmarks.
    """
    for start, end in edges:
        pt1 = landmarks[start] * scale
        pt2 = landmarks[end] * scale
        cv2.line(
            view,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            color,
            2
        )

def pose_estimation_3d_demo_with_projections(
    video_path: str,
    scale_factor: float = 1.0,
    show_mediapipe_overlay: bool = True
) -> None:
    """
    Demonstrates 3D pose estimation using MediaPipe Pose on a video,
    with three orthogonal projections of the normalized 3D keypoints.

    :param video_path: Path to the input video file.
    :param scale_factor: Factor by which to scale the frame for display.
    :param show_mediapipe_overlay: If True, use MediaPipe's own drawing function.
    :return: None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Define skeletal connections
    edges = [
        (23, 24), (23, 11), (24, 12),  # Hips to shoulders
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
        (11, 9), (9, 7),  # Left arm
        (12, 10), (10, 8),  # Right arm
    ]

    while True:
        time.sleep(0.03)  # For demonstration
        ret, frame = cap.read()
        if not ret:
            # End of video or read error
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            continue

        # Optionally resize for display
        if scale_factor != 1.0:
            disp_width = int(frame.shape[1] * scale_factor)
            disp_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Create blank images for orthogonal views
        height, width = 500, 500
        side_view = np.zeros((height, width, 3), dtype=np.uint8)
        front_view = np.zeros((height, width, 3), dtype=np.uint8)
        top_view = np.zeros((height, width, 3), dtype=np.uint8)

        # If a person is detected, extract keypoints
        if results.pose_landmarks:
            # Convert landmarks to numpy array
            landmarks_3d = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32
            )

            # Normalize the landmarks
            try:
                normalized_landmarks = normalize_pose_orientation(landmarks_3d)
            except ValueError as e:
                print(e)
                continue

            # Scale landmarks for the orthogonal views
            normalized_landmarks *= np.array([width / 2, height / 2, width / 2])
            normalized_landmarks += np.array([width / 2, height / 2, width / 2])

            # Plot side view (X-Z plane)
            for lm in normalized_landmarks:
                x, _, z = lm
                cv2.circle(side_view, (int(x), int(z)), 5, (0, 255, 0), -1)
            draw_connections(normalized_landmarks, side_view, edges, (0, 255, 0))

            # Plot front view (X-Y plane)
            for lm in normalized_landmarks:
                x, y, _ = lm
                cv2.circle(front_view, (int(x), int(y)), 5, (255, 0, 0), -1)
            draw_connections(normalized_landmarks, front_view, edges, (255, 0, 0))

            # Plot top view (Y-Z plane)
            for lm in normalized_landmarks:
                _, y, z = lm
                cv2.circle(top_view, (int(y), int(z)), 5, (0, 0, 255), -1)
            draw_connections(normalized_landmarks, top_view, edges, (0, 0, 255))

        # Display the main frame and orthogonal views
        cv2.imshow("3D Pose Estimation (MediaPipe)", frame)
        cv2.imshow("Side View (X-Z)", side_view)
        cv2.imshow("Front View (X-Y)", front_view)
        cv2.imshow("Top View (Y-Z)", top_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

def main():
    """
    Main function that runs the 3D pose estimation with orthogonal projections on a sample video.
    """
    # Replace with your own video file or use your webcam with 0
    video_path = "./data/deadlift_diagonal_view.mp4"

    pose_estimation_3d_demo_with_projections(
        video_path=video_path,
        scale_factor=1.0,
        show_mediapipe_overlay=True
    )

if __name__ == "__main__":
    main()
