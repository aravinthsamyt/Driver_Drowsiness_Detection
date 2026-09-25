import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    hinge_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib


# 1. Configuration
DATASET_DIR = "/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train"
IMG_SIZE = (64, 64)

categories = {"Closed": 0, "Open": 1}


# 2. Load and preprocess images
def load_eye_data(train_dir):
    data = []
    labels = []

    for category, label in categories.items():
        folder = os.path.join(train_dir, category)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)

            try:
                # Read image as grayscale
                img = imread(filepath, as_gray=True)

                # Resize to 64 × 64
                img = resize(img, IMG_SIZE)

                # Flatten: 64 × 64 → 4096 features
                img = img.flatten()

                data.append(img)
                labels.append(label)

            except Exception as e:
                print(f"Skipping {filename}: {e}")

    return np.array(data), np.array(labels)


X, y = load_eye_data(DATASET_DIR)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Closed images:", np.sum(y == 0))
print("Open images:", np.sum(y == 1))


# 3. Split: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# 4. Create a PURE SVM classifier
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )
)


# 5. Train the SVM
model.fit(X_train, y_train)

print("\nSVM training completed!")


# 6. Predictions
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_prob = model.predict_proba(X_train)
val_prob = model.predict_proba(X_val)
test_prob = model.predict_proba(X_test)


# 7. Calculate accuracy
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)


# 8. Calculate losses
# Hinge loss is the SVM's margin-based loss.
# Log loss is included for probability evaluation.

train_hinge = hinge_loss(
    np.where(y_train == 1, 1, -1),
    model.decision_function(X_train)
)

val_hinge = hinge_loss(
    np.where(y_val == 1, 1, -1),
    model.decision_function(X_val)
)

train_logloss = log_loss(y_train, train_prob, labels=[0, 1])
val_logloss = log_loss(y_val, val_prob, labels=[0, 1])
test_logloss = log_loss(y_test, test_prob, labels=[0, 1])


# 9. Print final report
print("\n========== SVM MODEL REPORT ==========")

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print(f"\nTrain Hinge Loss: {train_hinge:.4f}")
print(f"Validation Hinge Loss: {val_hinge:.4f}")

print(f"\nTrain Log Loss: {train_logloss:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")
print(f"Test Log Loss: {test_logloss:.4f}")


# 10. Accuracy and loss graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy graph
axes[0].bar(
    ["Train", "Validation", "Test"],
    [train_acc, val_acc, test_acc]
)
axes[0].set_title("SVM Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1.05)

# Hinge loss graph
axes[1].bar(
    ["Train", "Validation"],
    [train_hinge, val_hinge]
)
axes[1].set_title("SVM Hinge Loss")
axes[1].set_ylabel("Hinge Loss")

plt.tight_layout()
plt.show()


# 11. Confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Closed", "Open"]
)

disp.plot(cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()


# 12. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Closed", "Open"],
        zero_division=0
    )
)


# 13. Save the trained SVM
joblib.dump(model, "eye_svm_model.pkl")

print("\nModel saved as eye_svm_model.pkl")