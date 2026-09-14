import cv2
import mediapipe as mp
import math


# ============================================================
# MediaPipe Face Mesh
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Eye landmark indexes
# ============================================================

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]


# ============================================================
# Calculate EAR
# ============================================================

def calculate_ear(
    eye_landmarks,
    face_landmarks,
    frame_width,
    frame_height
):

    points = []

    for index in eye_landmarks:

        landmark = face_landmarks.landmark[index]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ============================================================
# Get eye bounding box
# ============================================================

def get_eye_bbox(
    eye_landmarks,
    face_landmarks,
    frame_width,
    frame_height,
    padding=10
):

    points = []

    for index in eye_landmarks:

        landmark = face_landmarks.landmark[index]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        points.append((x, y))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = max(0, min(xs) - padding)
    x_max = min(frame_width, max(xs) + padding)

    y_min = max(0, min(ys) - padding)
    y_max = min(frame_height, max(ys) + padding)

    return x_min, y_min, x_max, y_max


# ============================================================
# Crop eye image
# ============================================================

def crop_eye(
    frame,
    eye_landmarks,
    face_landmarks,
    padding=15
):

    height, width, _ = frame.shape

    x_min, y_min, x_max, y_max = get_eye_bbox(
        eye_landmarks,
        face_landmarks,
        width,
        height,
        padding
    )

    eye = frame[
        y_min:y_max,
        x_min:x_max
    ]

    return eye, (x_min, y_min, x_max, y_max)