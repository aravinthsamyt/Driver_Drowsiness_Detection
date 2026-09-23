import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report

# 1. Load Data (Using your structure)
def load_eye_data(train_dir, img_size=(64, 64)):
    data, labels = [], []
    categories = {'Closed': 0, 'Open': 1} # Corrected: Changed 'close' to 'Closed' and 'open' to 'Open'
    for category, label in categories.items():
        path = os.path.join(train_dir, category)
        if not os.path.exists(path):
            print(f"Warning: Folder not found at {path}") # Added a warning for clarity
            continue
        for img_name in os.listdir(path):
            try:
                img = imread(os.path.join(path, img_name), as_gray=True)
                data.append(resize(img, img_size).flatten())
                labels.append(label)
            except:
                continue
    return np.array(data), np.array(labels)

# Load dataset
train_directory = '/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train'
X, y = load_eye_data(train_directory)

# Split into Train (70%), Validation (15%), and Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp) # Corrected stratify parameter

# 2. Iterative Training Loop to Track Metrics
# Using log_loss enables true probability outputs to track validation loss precisely
model = SGDClassifier(loss='log_loss', alpha=0.001, random_state=42)
classes = np.unique(y_train)

epochs = 50
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

print("Training model and tracking metrics...")
for epoch in range(epochs):
    # Train for one partial epoch iteration
    model.partial_fit(X_train, y_train, classes=classes)

    # Predict probabilities for loss calculations
    train_prob = model.predict_proba(X_train)
    val_prob = model.predict_proba(X_val)

    # Calculate hard predictions for accuracy calculations
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Log values to history dictionary
    history['train_loss'].append(log_loss(y_train, train_prob))
    history['val_loss'].append(log_loss(y_val, val_prob))
    history['train_acc'].append(accuracy_score(y_train, train_pred))
    history['val_acc'].append(accuracy_score(y_val, val_pred))

print("Training complete!\n")

# 3. Final Test Evaluation
test_prob = model.predict_proba(X_test)
test_pred = model.predict(X_test)

final_train_acc = history['train_acc'][-1]
final_train_loss = history['train_loss'][-1]
final_val_acc = history['val_acc'][-1]
final_val_loss = history['val_loss'][-1]
final_test_acc = accuracy_score(y_test, test_pred)
final_test_loss = log_loss(y_test, test_prob)

# 4. Print Formatted Textual Report
print("="*35)
print("       FINAL MODEL REPORT       ")
print("="*35)
print(f"train accuracy: {final_train_acc:.6f}, train loss: {final_train_loss:.6f}")
print(f"val accuracy:   {final_val_acc:.6f}, val loss:   {final_val_loss:.6f}")
print(f"test accuracy:  {final_test_acc:.6f}, test loss:  {final_test_loss:.6f}")
print("="*35)

# 5. Plot Accuracy & Loss Evaluation Graphs
plt.figure(figsize=(14, 5))

# Plot 1: Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
plt.axhline(y=final_test_acc, color='green', linestyle='--', label=f'Test Accuracy ({final_test_acc:.2f})')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.axhline(y=final_test_loss, color='green', linestyle='--', label=f'Test Loss ({final_test_loss:.2f})')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()