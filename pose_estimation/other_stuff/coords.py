import numpy as np
import matplotlib.pyplot as plt


# Example realistic landmarks for a standing person (simplified and exaggerated for demonstration)
# MediaPipe format: [x, y, z, visibility]
example_landmarks = np.array(
    [
        [0.5, 0.8, 0.0, 1.0],  # Nose
        [0.45, 0.75, -0.02, 1.0],  # Left Eye
        [0.55, 0.75, -0.02, 1.0],  # Right Eye
        [0.4, 0.72, -0.04, 1.0],  # Left Ear
        [0.6, 0.72, -0.04, 1.0],  # Right Ear
        [0.45, 0.6, -0.05, 1.0],  # Left Shoulder
        [0.55, 0.6, -0.05, 1.0],  # Right Shoulder
        [0.4, 0.5, -0.05, 1.0],  # Left Elbow
        [0.6, 0.5, -0.05, 1.0],  # Right Elbow
        [0.35, 0.4, -0.05, 1.0],  # Left Wrist
        [0.65, 0.4, -0.05, 1.0],  # Right Wrist
        [0.48, 0.4, 0.0, 1.0],  # Left Hip
        [0.52, 0.4, 0.0, 1.0],  # Right Hip
        [0.45, 0.2, 0.0, 1.0],  # Left Knee
        [0.55, 0.2, 0.0, 1.0],  # Right Knee
        [0.43, 0.0, 0.02, 1.0],  # Left Ankle
        [0.57, 0.0, 0.02, 1.0],  # Right Ankle
    ]
)

# Reference distance (e.g., measured hip distance in meters)
reference_distance = 0.05  # 30 cm


def normalize_pose_orientation(
    landmarks_3d: np.ndarray,
    reference_distance: float,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """
    Normalizes the pose's orientation by aligning the torso to a canonical frame.
    The hips are set at the origin, and the torso's orientation is aligned to a standard axis.

    :param landmarks_3d: Numpy array of shape (33, 4) with normalized (x, y, z, visibility).
    :param reference_distance: Real-world distance in meters (e.g., distance between hips).
    :param confidence_threshold: Minimum visibility to include a landmark.
    :return: Numpy array of shape (33, 3) with (X, Y, Z) in real-world units (meters),
             normalized to a canonical orientation.
    """
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    left_hip = landmarks_3d[LEFT_HIP]
    right_hip = landmarks_3d[RIGHT_HIP]
    left_shoulder = landmarks_3d[LEFT_SHOULDER]
    right_shoulder = landmarks_3d[RIGHT_SHOULDER]

    if left_hip[3] < confidence_threshold or right_hip[3] < confidence_threshold:
        raise ValueError(
            "Low confidence in hip landmarks, cannot normalize orientation."
        )

    hip_center = (left_hip[:3] + right_hip[:3]) / 2
    shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2
    torso_y = shoulder_center - hip_center
    torso_y /= np.linalg.norm(torso_y)
    hip_line = right_hip[:3] - left_hip[:3]
    torso_x = np.cross(hip_line, torso_y)
    torso_x /= np.linalg.norm(torso_x)
    torso_z = np.cross(torso_x, torso_y)
    torso_z /= np.linalg.norm(torso_z)
    rotation_matrix = np.vstack([torso_x, torso_y, torso_z]).T

    norm_hip_dist = np.linalg.norm(left_hip[:3] - right_hip[:3])
    scale_factor = reference_distance / norm_hip_dist

    absolute_landmarks = np.zeros((landmarks_3d.shape[0], 3), dtype=np.float32)
    for i, (nx, ny, nz, visibility) in enumerate(landmarks_3d):
        if visibility < confidence_threshold:
            continue

        position = np.array([nx, ny, nz]) * scale_factor
        position -= hip_center
        position = rotation_matrix @ position
        absolute_landmarks[i] = position

    return absolute_landmarks


# Normalize the example landmarks
normalized_landmarks = normalize_pose_orientation(example_landmarks, reference_distance)


# Function to plot 3D landmarks
def plot_landmarks_3d(landmarks, title, ax):
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c="r", marker="o")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])


# Define skeletal connections for keypoints (edges to connect landmarks)
edges = [
    (11, 12),
    (11, 5),
    (12, 6),  # Hips to shoulders
    (5, 6),  # Shoulders
    (5, 7),
    (7, 9),  # Left arm
    (6, 8),
    (8, 10),  # Right arm
    (11, 13),
    (13, 15),  # Left leg
    (12, 14),
    (14, 16),  # Right leg
]


# Function to plot 3D landmarks with connections
def plot_landmarks_with_connections(landmarks, title, ax, edges):
    """
    Plots 3D landmarks with connections (edges).

    :param landmarks: Numpy array of shape (N, 3).
    :param title: Title of the plot.
    :param ax: Matplotlib 3D axis.
    :param edges: List of tuples representing connections between keypoints.
    """
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        landmarks[:, 2],
        c="r",
        marker="o",
        label="Keypoints",
    )
    for edge in edges:
        start, end = edge
        ax.plot(
            [landmarks[start, 0], landmarks[end, 0]],
            [landmarks[start, 1], landmarks[end, 1]],
            [landmarks[start, 2], landmarks[end, 2]],
            c="b",
            label="Connections" if edge == edges[0] else None,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])


# Plot original and normalized landmarks with connections
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

plot_landmarks_with_connections(
    example_landmarks[:, :3], "Original Landmarks", ax1, edges
)
plot_landmarks_with_connections(
    normalized_landmarks, "Normalized Landmarks", ax2, edges
)

plt.tight_layout()
plt.show()

# Plot normalized landmarks from 3 orthogonal views
fig = plt.figure(figsize=(18, 6))

# Define the orthogonal views
view_angles = [
    (90, 0),  # Side view (X-Z plane)
    (0, 0),  # Front view (X-Y plane)
    (0, 90),  # Top view (Y-Z plane)
]

for i, (elev, azim) in enumerate(view_angles, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    plot_landmarks_with_connections(normalized_landmarks, f"View {i}", ax, edges)
    ax.view_init(elev=elev, azim=azim)  # Set the viewing angles

plt.tight_layout()
plt.show()
