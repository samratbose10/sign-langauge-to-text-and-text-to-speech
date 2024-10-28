import tensorflow as tf
from keras import layers
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

DATASET_DIR = 'dataset'
IMAGE_SIZE = (64, 64)

def load_images_and_landmarks(dataset_path, image_size=(64, 64)):
    X_images, X_landmarks, y = [], [], []
    letters = os.listdir(dataset_path)

    for letter in letters:
        img_folder = os.path.join(dataset_path, letter)
        for img_file in os.listdir(img_folder):
            if img_file.endswith('.png'):
                img_path = os.path.join(img_folder, img_file)
                txt_path = img_path.replace('.png', '.txt')

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, image_size)
                img = img / 255.0  
                X_images.append(img)

                if os.path.exists(txt_path):
                    landmarks = np.loadtxt(txt_path)  
                    X_landmarks.append(landmarks.flatten())  

        
                y.append(letter)

    return np.array(X_images), np.array(X_landmarks), np.array(y)

X_images, X_landmarks, y = load_images_and_landmarks(DATASET_DIR)

X_images = X_images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

label_binarizer = LabelBinarizer()
y_bin = label_binarizer.fit_transform(y)

X_train_images, X_val_images, X_train_landmarks, X_val_landmarks, y_train, y_val = train_test_split(
    X_images, X_landmarks, y_bin, test_size=0.2, random_state=42)

def build_multi_input_model(image_shape=(64, 64, 1), landmark_shape=(42,), num_classes=26):

    image_input = keras.Input(shape=image_shape, name="image_input")
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Flatten()(x1)
    landmark_input = keras.Input(shape=landmark_shape, name="landmark_input")
    x2 = layers.Dense(128, activation='relu')(landmark_input)
    x2 = layers.Dense(64, activation='relu')(x2)
    concatenated = layers.concatenate([x1, x2])
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=[image_input, landmark_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_multi_input_model()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit([X_train_images, X_train_landmarks], y_train, epochs=20, batch_size=32,
                    validation_data=([X_val_images, X_val_landmarks], y_val),
                    callbacks=[early_stopping, model_checkpoint])

val_loss, val_acc = model.evaluate([X_val_images, X_val_landmarks], y_val)
print(f"Validation accuracy: {val_acc * 100:.2f}%")
