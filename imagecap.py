import cv2
import os
import mediapipe as mp
import numpy as np

DATASET_DIR = 'dataset'
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for letter in letters:
    os.makedirs(os.path.join(DATASET_DIR, letter), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)

print("Press the key corresponding to the letter you want to capture images for (A-Z).")
print("Press esc to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.Exiting...")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                landmark_list.append([int(lm.x * w), int(lm.y * h)])

    cv2.imshow('Hand Gesture Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if chr(key).upper() in letters:
        letter = chr(key).upper()

        img_save_path = os.path.join(DATASET_DIR, letter, f'{letter}_{len(os.listdir(os.path.join(DATASET_DIR, letter)))}.png')
        landmark_save_path = os.path.join(DATASET_DIR, letter, f'{letter}_{len(os.listdir(os.path.join(DATASET_DIR, letter)))}.txt')

        cv2.imwrite(img_save_path, frame)
        print(f"Saved image for letter '{letter}' at {img_save_path}")

        np.savetxt(landmark_save_path, landmark_list, fmt='%d')
        print(f"Saved landmarks for letter '{letter}' at {landmark_save_path}")

cap.release()
cv2.destroyAllWindows()
