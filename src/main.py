import cv2
import tensorflow as tf
import numpy as np
import os

from face_mesh import (
    face_mesh,
    LEFT_EYE,
    RIGHT_EYE,
    calculate_ear,
    crop_eye
)


# ============================================================
# 1. Load CNN model
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    "model",
    "best_eye_model.keras"
)

model = tf.keras.models.load_model(
    model_path
)

print("✅ CNN model loaded successfully!")

print(
    "Model input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# 2. Get CNN input size
# ============================================================

IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]

print(
    f"Eye image size: "
    f"{IMG_WIDTH} x {IMG_HEIGHT}"
)


# ============================================================
# 3. CNN prediction function
# ============================================================

def predict_eye(eye_image):

    # Check whether eye was detected
    if eye_image is None:
        return None, 0.0

    if eye_image.size == 0:
        return None, 0.0

    # Resize to CNN input size
    eye_image = cv2.resize(
        eye_image,
        (IMG_WIDTH, IMG_HEIGHT)
    )

    # BGR → RGB
    eye_image = cv2.cvtColor(
        eye_image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float
    eye_image = eye_image.astype(
        np.float32
    )

    # Normalize
    eye_image = eye_image / 255.0

    # Add batch dimension
    eye_image = np.expand_dims(
        eye_image,
        axis=0
    )

    # CNN prediction
    prediction = model.predict(
        eye_image,
        verbose=0
    )[0][0]

    # ----------------------------------------
    # IMPORTANT:
    # Assumes:
    # prediction >= 0.5 → OPEN
    # prediction < 0.5  → CLOSED
    # ----------------------------------------

    if prediction >= 0.5:

        label = "OPEN"

        confidence = prediction * 100

    else:

        label = "CLOSED"

        confidence = (
            1 - prediction
        ) * 100

    return label, confidence


# ============================================================
# 4. Open webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Cannot open webcam")
    exit()

print("📷 Webcam started")
print("Press Q to quit")


# ============================================================
# 5. Real-time loop
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("❌ Unable to access webcam")
        break


    # ----------------------------------------
    # Mirror webcam
    # ----------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # ----------------------------------------
    # Frame dimensions
    # ----------------------------------------

    height, width, _ = frame.shape


    # ----------------------------------------
    # BGR → RGB
    # ----------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------
    # MediaPipe Face Mesh
    # ----------------------------------------

    results = face_mesh.process(
        rgb
    )


    # Default values
    left_label = "N/A"
    right_label = "N/A"

    left_confidence = 0.0
    right_confidence = 0.0

    left_ear = 0.0
    right_ear = 0.0

    ear = 0.0

    left_eye = None
    right_eye = None


    # ========================================================
    # Face detected
    # ========================================================

    if results.multi_face_landmarks:

        face_landmarks = (
            results.multi_face_landmarks[0]
        )


        # ====================================================
        # Calculate EAR
        # ====================================================

        left_ear = calculate_ear(
            LEFT_EYE,
            face_landmarks,
            width,
            height
        )

        right_ear = calculate_ear(
            RIGHT_EYE,
            face_landmarks,
            width,
            height
        )

        ear = (
            left_ear +
            right_ear
        ) / 2.0


        # ====================================================
        # Crop left eye
        # ====================================================

        left_eye, left_box = crop_eye(
            frame,
            LEFT_EYE,
            face_landmarks,
            padding=15
        )


        # ====================================================
        # Crop right eye
        # ====================================================

        right_eye, right_box = crop_eye(
            frame,
            RIGHT_EYE,
            face_landmarks,
            padding=15
        )


        # ====================================================
        # CNN prediction - LEFT eye
        # ====================================================

        left_label, left_confidence = predict_eye(
            left_eye
        )


        # ====================================================
        # CNN prediction - RIGHT eye
        # ====================================================

        right_label, right_confidence = predict_eye(
            right_eye
        )


        # ====================================================
        # Draw eye bounding boxes
        # ====================================================

        lx1, ly1, lx2, ly2 = left_box

        rx1, ry1, rx2, ry2 = right_box


        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            (0, 255, 0),
            2
        )


        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 0),
            2
        )


    # ========================================================
    # Determine final eye status
    # ========================================================

    if (
        left_label == "CLOSED"
        and
        right_label == "CLOSED"
    ):

        cnn_status = "CLOSED"

    elif (
        left_label == "OPEN"
        and
        right_label == "OPEN"
    ):

        cnn_status = "OPEN"

    else:

        cnn_status = "MIXED"


    # ========================================================
    # EAR status
    # ========================================================

    EAR_THRESHOLD = 0.25

    if ear > 0:

        if ear < EAR_THRESHOLD:

            ear_status = "CLOSED"

        else:

            ear_status = "OPEN"

    else:

        ear_status = "NO FACE"


    # ========================================================
    # Display information
    # ========================================================

    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Left CNN: {left_label} "
        f"{left_confidence:.1f}%",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Right CNN: {right_label} "
        f"{right_confidence:.1f}%",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"CNN EYES: {cnn_status}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"EAR STATUS: {ear_status}",
        (30, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # ========================================================
    # Show webcam
    # ========================================================

    cv2.imshow(
        "Drowsiness Detection - Face Mesh + CNN",
        frame
    )


    # ========================================================
    # Quit
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# Release resources
# ============================================================

cap.release()

cv2.destroyAllWindows()