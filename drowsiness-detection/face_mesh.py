import cv2
import mediapipe as mp
import math


# --------------------------------
# Calculate EAR
# --------------------------------
def calculate_ear(eye_landmarks, face_landmarks, frame_width, frame_height):

    points = []

    for index in eye_landmarks:

        landmark = face_landmarks.landmark[index]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    # Vertical distances
    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    # Horizontal distance
    horizontal = math.dist(p1, p4)

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# --------------------------------
# MediaPipe Face Mesh
# --------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------
# Eye landmark indexes
# --------------------------------

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]


# --------------------------------
# Open webcam
# --------------------------------

cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        print("Unable to access webcam")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, _ = frame.shape

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    results = face_mesh.process(rgb)


    # --------------------------------
    # If face detected
    # --------------------------------

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]


        # Calculate LEFT eye EAR
        left_ear = calculate_ear(
            LEFT_EYE,
            face_landmarks,
            width,
            height
        )


        # Calculate RIGHT eye EAR
        right_ear = calculate_ear(
            RIGHT_EYE,
            face_landmarks,
            width,
            height
        )


        # Average both eyes
        ear = (left_ear + right_ear) / 2.0


        # --------------------------------
        # Display EAR
        # --------------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display individual eyes
        cv2.putText(
            frame,
            f"Left: {left_ear:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right: {right_ear:.2f}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    ear = (left_ear + right_ear) / 2.0
    EAR_THRESHOLD = 0.25

    if ear < EAR_THRESHOLD:
        status = "CLOSED"
    else:
        status = "OPEN"
    cv2.putText(
    frame,
    f"EYES: {status}",
    (30, 160),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
    )    
    # --------------------------------
    # Show webcam
    # --------------------------------

    cv2.imshow(
        "Drowsiness Detection - EAR",
        frame
    )
    
        # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------
# Release resources
# --------------------------------

cap.release()
cv2.destroyAllWindows()