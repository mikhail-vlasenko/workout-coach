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


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle in degrees between two vectors.

    :param v1: First vector (3D).
    :param v2: Second vector (3D).
    :return: Angle in degrees.
    """
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))
    return np.degrees(angle_rad)


def compute_torso_leg_angle(keypoints: np.ndarray) -> float:
    """
    Compute the angle between the torso and legs.

    :param keypoints: 3D pose keypoints.
    :return: Angle in degrees.
    """
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :3]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, :3]
    left_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value, :3]
    right_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value, :3]
    left_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value, :3]
    right_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value, :3]

    torso_vector = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
    leg_vector = (left_knee + right_knee) / 2 - (left_hip + right_hip) / 2

    return calculate_angle(torso_vector, leg_vector)


def compute_knee_angle(keypoints: np.ndarray, side: str = "left") -> float:
    """
    Compute the knee angle.

    :param keypoints: 3D pose keypoints.
    :param side: "left" or "right" knee.
    :return: Knee angle in degrees.
    """
    hip = keypoints[mp_pose.PoseLandmark[f"{side.upper()}_HIP".format()].value, :3]
    knee = keypoints[mp_pose.PoseLandmark[f"{side.upper()}_KNEE".format()].value, :3]
    ankle = keypoints[mp_pose.PoseLandmark[f"{side.upper()}_ANKLE".format()].value, :3]

    thigh_vector = hip - knee
    calf_vector = ankle - knee

    return calculate_angle(thigh_vector, calf_vector)


def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two points.

    :param point1: First point (3D).
    :param point2: Second point (3D).
    :return: Distance.
    """
    return np.linalg.norm(point1 - point2)


def are_arms_below_knees(keypoints: np.ndarray) -> bool:
    """
    Determine if arms are below knees based on x-coordinates.

    :param keypoints: 3D pose keypoints.
    :return: True if arms are below, False otherwise.
    """
    left_wrist_x = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value, 1]
    right_wrist_x = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value, 1]
    left_knee_x = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value, 1]
    right_knee_x = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value, 1]

    return (
        left_wrist_x > left_knee_x and right_wrist_x > right_knee_x
    )  # For front view; adjust as needed for other views.


def analyze_pose(keypoints: np.ndarray, state: dict) -> dict:
    """
    Perform analysis on pose keypoints and track repetitions with timing.

    :param keypoints: 3D pose keypoints.
    :param state: A dictionary to store state information (e.g., rep count and timing).
    :return: Dictionary with computed metrics and updated state.
    """
    if keypoints.size == 0:
        return {}

    current_time = time.time()
    arms_below_knees = are_arms_below_knees(keypoints)

    # Check for rep transitions
    if state.get("below_knees") and not arms_below_knees:
        state["rep_count"] += 1
        if state["last_rep_time"] is not None:
            state["rep_time"] = current_time - state["last_rep_time"]
        state["last_rep_time"] = current_time

    state["below_knees"] = arms_below_knees

    results = {
        "torso_leg_angle": compute_torso_leg_angle(keypoints),
        "left_knee_angle": compute_knee_angle(keypoints, side="left"),
        "right_knee_angle": compute_knee_angle(keypoints, side="right"),
        "feet_distance": compute_distance(
            keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value, :3],
            keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value, :3],
        ),
        "shoulder_distance": compute_distance(
            keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value, :3],
            keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value, :3],
        ),
        "are_arms_below_knees": arms_below_knees,
        "rep_count": state["rep_count"],
        "rep_time": state.get("rep_time", None),  # Time for the last rep
        "grip_width": compute_distance(
            keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value, :3],
            keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value, :3],
        ),
        "hands_outside_knees": keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value, 0] > keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value, 0] and keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value, 0] < keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value, 0],
    }

    return results

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
        min_tracking_confidence=0.5
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
    frame: np.ndarray,
    landmarks_3d: np.ndarray,
    confidence_threshold: float = 0.5
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
    video_path: str,
    scale_factor: float = 1.0,
    show_mediapipe_overlay: bool = True
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
        min_tracking_confidence=0.5
    )

    # Initialize state for rep counting and timing
    state = {"rep_count": 0, "below_knees": False, "last_rep_time": None, "rep_time": None}

    while True:
        # time.sleep(0.5)

        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Optionally resize for display
        if scale_factor != 1.0:
            disp_width = int(frame.shape[1] * scale_factor)
            disp_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (disp_width, disp_height), interpolation=cv2.INTER_AREA)

        # If a person is detected, extract keypoints
        if results.pose_landmarks:
            # Convert landmarks to numpy array
            landmarks_3d = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32
            )

            # Analyze pose and update state
            analysis_results = analyze_pose(landmarks_3d, state)

            # Print results to console
            print(analysis_results)

            # Display rep count on the frame
            rep_time_display = (
                f"Last Rep Time: {state['rep_time']:.2f}s" if state["rep_time"] else "Last Rep Time: N/A"
            )
            cv2.putText(
                frame,
                f"Reps: {state['rep_count']} | {rep_time_display}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Draw the keypoints (optional visualization)
            if show_mediapipe_overlay:
                disp_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    disp_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                frame = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
            else:
                draw_keypoints_2d(frame, landmarks_3d, confidence_threshold=0.5)

        # Show the frame with visualization
        cv2.imshow("3D Pose Estimation (MediaPipe)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


def main():
    """
    Main function that runs the 3D pose estimation on a sample video.
    """
    # Replace with your own video file or use your webcam with 0
    video_path = "./data/deadlift_diagonal.mp4"

    pose_estimation_3d_demo(
        video_path=video_path,
        scale_factor=1.0,
        show_mediapipe_overlay=True
    )


if __name__ == "__main__":
    main()
