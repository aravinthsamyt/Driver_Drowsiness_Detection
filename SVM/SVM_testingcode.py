import os
import numpy as np
import joblib

from skimage.io import imread
from skimage.transform import resize

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
#hello im testing the code
import matplotlib.pyplot as plt


# ==========================================
# 1. Load trained SVM model
# ==========================================

model = joblib.load("eye_svm_model.pkl")

print("SVM model loaded successfully!")


# ==========================================
# 2. New Dataset path
# ==========================================

DATASET_DIR = "/content/drive/MyDrive/computer_vision2026/Mini_project/new_Dataset"

IMG_SIZE = (64, 64)


# Actual labels
# close = 0
# open  = 1

categories = {
    "close": 0,
    "open": 1
}


# ==========================================
# 3. Load unseen test images
# ==========================================

X_test = []
y_test = []
image_names = []

for category, label in categories.items():

    folder = os.path.join(DATASET_DIR, category)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        try:

            # Read image as grayscale
            img = imread(filepath, as_gray=True)

            # Resize exactly like training
            img = resize(img, IMG_SIZE)

            # Flatten 64 × 64 → 4096
            img = img.flatten()

            X_test.append(img)
            y_test.append(label)
            image_names.append(filename)

        except Exception as e:

            print("Error:", filename, e)


X_test = np.array(X_test)
y_test = np.array(y_test)


print("\n================================")
print("NEW DATASET INFORMATION")
print("================================")

print("Total images :", len(X_test))
print("Open images  :", np.sum(y_test == 1))
print("Close images :", np.sum(y_test == 0))


# ==========================================
# 4. Predict using SVM
# ==========================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)


# ==========================================
# 5. Display every image prediction
# ==========================================

class_names = {
    0: "CLOSED",
    1: "OPEN"
}


print("\n================================")
print("IMAGE-WISE PREDICTIONS")
print("================================")

for i in range(len(X_test)):

    actual = class_names[y_test[i]]

    predicted = class_names[predictions[i]]

    confidence = (
        probabilities[i][list(model.classes_).index(predictions[i])]
        * 100
    )

    result = "CORRECT" if y_test[i] == predictions[i] else "WRONG"

    print(
        f"{image_names[i]:25s} | "
        f"Actual: {actual:6s} | "
        f"Predicted: {predicted:6s} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{result}"
    )


# ==========================================
# 6. Overall accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("FINAL TEST RESULT")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 7. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Closed", "Open"]
)

disp.plot(cmap="Blues")

plt.title("SVM - New Dataset Confusion Matrix")
plt.show()


# ==========================================
# 8. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Closed", "Open"],
        zero_division=0
    )
)