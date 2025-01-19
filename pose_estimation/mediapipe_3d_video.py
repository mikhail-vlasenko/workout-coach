import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))
    return np.degrees(angle_rad)


def are_arms_below_knees(keypoints: np.ndarray) -> bool:
    left_wrist_y = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value, 1]
    right_wrist_y = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value, 1]
    left_knee_y = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value, 1]
    right_knee_y = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value, 1]

    return left_wrist_y > left_knee_y and right_wrist_y > right_knee_y


def analyze_pose(keypoints: np.ndarray, state: dict) -> dict:
    if keypoints.size == 0:
        return {}

    arms_below_knees = are_arms_below_knees(keypoints)

    if arms_below_knees and not state["rep_in_progress"]:
        state["rep_start_time"] = time.time()
        state["rep_in_progress"] = True
    elif not arms_below_knees and state["rep_in_progress"]:
        rep_time = time.time() - state["rep_start_time"]
        state["rep_count"] += 1
        state["rep_in_progress"] = False
        state["last_rep_time"] = rep_time

    return {
        "rep_count": state["rep_count"],
        "last_rep_time": state.get("last_rep_time", None),
    }


def detect_keypoints_mediapipe_3d(frame: np.ndarray) -> np.ndarray:
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


def draw_overlay(frame: np.ndarray, state: dict) -> None:
    # Assuming frame and state are already defined
    h, w, _ = frame.shape
    rep_count_text = f"Reps: {state['rep_count']}"
    last_rep_time_text = (
        f"Rep Time: {state.get('last_rep_time', 'N/A'):.2f} s"
        if state.get('last_rep_time')
        else "Rep Time: -"
    )

    # Define new text properties
    font = cv2.FONT_HERSHEY_COMPLEX  # Nicer font
    font_scale = 1.2  # Smaller font size
    font_color = (255, 255, 255)  # Black text for better readability
    thickness = 2  # Thinner text
    line_type = cv2.LINE_AA  # Anti-aliased for smoother text

    # Background rectangle properties
    bg_color = (100, 200, 100)  # White background
    alpha = 0.6  # Transparency factor

    # Determine text sizes
    rep_count_size = cv2.getTextSize(rep_count_text, font, font_scale, thickness)[0]
    last_rep_time_size = cv2.getTextSize(last_rep_time_text, font, font_scale, thickness)[0]

    # Define text positions and rectangle coordinates
    padding = 20
    x, y = 60, 60  # Starting position for the first text
    rect_width = max(rep_count_size[0], last_rep_time_size[0]) + 2 * padding
    rect_height = rep_count_size[1] + last_rep_time_size[1] + 3 * padding

    y = h - rect_height - padding  # Position the text at the bottom of the frame

    # Create a translucent overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - padding, y - padding),
        (x - padding + rect_width, y + rect_height - padding),
        bg_color,
        -1,
    )

    # Blend the overlay with the frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw text over the translucent rectangle
    cv2.putText(frame, rep_count_text, (x, y + rep_count_size[1]), font, font_scale, font_color, thickness, line_type)
    cv2.putText(frame, last_rep_time_text, (x, y + 2*rep_count_size[1] + padding), font, font_scale, font_color,
                thickness, line_type)

def post_estimation_3d_from_frame(bgr_frame, pose_api, state, show_mediapipe_overlay):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    results = pose_api.process(rgb_frame)

    if results.pose_landmarks:
        landmarks_3d = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

        analysis_results = analyze_pose(landmarks_3d, state)

        if show_mediapipe_overlay:
            # Define the drawing specifications for green landmarks and connections
            landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            # Drawing the landmarks with the specified styles
            mp_drawing.draw_landmarks(
                bgr_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec,
            )

        draw_overlay(bgr_frame, state)

    return bgr_frame


def pose_estimation_3d_from_video(video_path: str, show_mediapipe_overlay: bool = True) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    pose_api = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    state = {
        "rep_count": 0,
        "rep_in_progress": False,
        "rep_start_time": None,
        "last_rep_time": None,
    }
    loop = 0
    video = []

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            loop += 1
            cap = cv2.VideoCapture(video_path)
            ret, bgr_frame = cap.read()


        bgr_frame_viz = post_estimation_3d_from_frame(
            bgr_frame, pose_api, state, show_mediapipe_overlay
        )
        video.append(bgr_frame_viz)

        if loop >= 1:
            save_frames_to_video(video, "output2_green.mp4")
            break

        cv2.imshow("3D Pose Estimation (MediaPipe)", bgr_frame_viz)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



    cap.release()
    pose_api.close()
    cv2.destroyAllWindows()


import cv2


def save_frames_to_video(frames, output_path, fps=30):
    """
    Save a list of frames as an MP4 video.

    Parameters:
    - frames (list of np.ndarray): List of frames to save as video.
    - output_path (str): Path to save the output video file (e.g., "output.mp4").
    - fps (int): Frames per second for the output video.
    """
    if not frames:
        raise ValueError("The frames list is empty.")

    # Get the frame height, width, and initialize VideoWriter
    height, width, _ = frames[0].shape
    # Save video frames as WebM file
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved successfully to {output_path}")


def main():
    filename = "vid2"
    video_path = f"../data/{filename}.mp4"
    pose_estimation_3d_from_video(video_path=video_path, show_mediapipe_overlay=True)


if __name__ == "__main__":
    main()
