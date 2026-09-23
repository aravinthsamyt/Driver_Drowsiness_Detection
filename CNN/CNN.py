import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train"
print(os.listdir("/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train"))

open_path = os.path.join(dataset_path, "open")
closed_path = os.path.join(dataset_path, "closed")

print("Open images:", len(os.listdir("/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train/Open")))
print("Closed images:", len(os.listdir("/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train/Closed")))

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(dataset.class_names)

#Folder splitting into train, validation and test

import os
import random
import shutil

dataset_path = "/content/drive/MyDrive/computer_vision2026/Mini_project/Dataset/train"
output_path = "/content/eye_dataset"

random.seed(42)

classes = ["Open", "Closed"]

# Create Train / Validation / Test folders
for split in ["train", "validation", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(output_path, split, class_name),
            exist_ok=True
        )

for class_name in classes:

    source_folder = os.path.join(dataset_path, class_name)

    files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Randomly shuffle
    random.shuffle(files)

    total = len(files)

    # 80% Train, 10% Validation, 10% Test
    train_end = int(total * 0.80)
    validation_end = int(total * 0.90)

    train_files = files[:train_end]
    validation_files = files[train_end:validation_end]
    test_files = files[validation_end:]

    # Copy Train
    for file in train_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(output_path, "train", class_name, file)
        )

    # Copy Validation
    for file in validation_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(output_path, "validation", class_name, file)
        )

    # Copy Test
    for file in test_files:
        shutil.copy2(
            os.path.join(source_folder, file),
            os.path.join(output_path, "test", class_name, file)
        )

    print(f"\n{class_name}")
    print("Total      :", total)
    print("Train      :", len(train_files))
    print("Validation :", len(validation_files))
    print("Test       :", len(test_files))

print("\nDataset split completed!")


import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_dir = os.path.join(output_path, 'train')
validation_dir = os.path.join(output_path, 'validation')
test_dir = os.path.join(output_path, 'test')

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_dataset.class_names)


normalization_layer = tf.keras.layers.Rescaling(1./255)

#CNN Architecture

model = tf.keras.Sequential([

    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "/content/drive/MyDrive/drowsiness/best_eye_model.keras",
    monitor='val_loss',
    save_best_only=True
)

model.summary()

#Training the model

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[early_stopping, checkpoint]
)

best_model = tf.keras.models.load_model(
    "/content/drive/MyDrive/drowsiness/best_eye_model.keras"
)

#model evaluation on validation dataset

loss, accuracy = best_model.evaluate(validation_dataset)

print("Best Model Validation Loss:", loss)
print("Best Model Validation Accuracy:", accuracy)

#training and validation accuracy

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training and validation loss

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Train,Validation and Test accuracy

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.axhline(test_accuracy, linestyle='--', label=f'Test Accuracy ({test_accuracy:.2f})')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training, Validation and Test loss

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.axhline(test_loss, linestyle='--',
            label=f'Test Loss ({test_loss:.2f})')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#````````````````````````````````````````````````````````````````
#Confusion Matrix and Classification Report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)

print(cm)

class_names = validation_dataset.class_names

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Eye CNN')
plt.show()

#classificatio report

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

#model loaded or save

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model("/content/drive/MyDrive/drowsiness/best_eye_model.keras")

print("Model loaded successfully!")
dataset_path = "/content/drive/MyDrive/computer_vision2026/Mini_project/new_Dataset"

IMG_SIZE = (128, 128) # Changed from (224, 224) to match model's training input size
BATCH_SIZE = 32

new_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class names:", new_dataset.class_names)




#Finaly find unseen new images

# Get image file paths
image_paths = []

for class_name in new_dataset.class_names:
    class_folder = os.path.join(dataset_path, class_name)

    for filename in sorted(os.listdir(class_folder)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(
                (os.path.join(class_folder, filename), class_name)
            )

print("Total image files:", len(image_paths))

plt.figure(figsize=(15, 20))

for i, (image_path, actual_class) in enumerate(image_paths):

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "open"
        confidence = prediction * 100
    else:
        predicted_class = "close"
        confidence = (1 - prediction) * 100

    # Image
    plt.subplot(7, 7, i + 1)
    plt.imshow(img)
    plt.axis("off")

    # Mark correct/wrong
    if actual_class == predicted_class:
        status = "✓"
    else:
        status = "✗"

    plt.title(
        f"{status}\n"
        f"Actual: {actual_class}\n"
        f"Pred: {predicted_class}\n"
        f"{confidence:.1f}%",
        fontsize=9
    )

plt.tight_layout()
plt.show()