import tensorflow as tf
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3 

engine = pyttsx3.init()

engine.setProperty('rate', 150)  
engine.setProperty('volume', 1.0) 

voices = engine.getProperty('voices')
for voice in voices:
    if 'india' in voice.id.lower():  
        engine.setProperty('voice', voice.id)
        break

def speak_message(message):
    engine.say(message)
    engine.runAndWait()

speak_message("Namaste, Apka Swagat Hai")

model = load_model('best_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) 
mp_drawing = mp.solutions.drawing_utils  

IMAGE_SIZE = (64, 64)

previous_prediction = None
start_time = None
word = "" 

def preprocess_image(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    frame_resized = cv2.resize(frame_gray, IMAGE_SIZE)  
    frame_normalized = frame_resized / 255.0  
    frame_reshaped = np.reshape(frame_normalized, (1, 64, 64, 1))  
    return frame_reshaped

def preprocess_landmarks(landmarks):
    landmark_array = []
    for lm in landmarks.landmark:
        landmark_array.append([lm.x, lm.y])
    landmark_array = np.array(landmark_array).flatten()  
    landmark_array = np.reshape(landmark_array, (1, -1)) 
    return landmark_array

def speak_word(word):
    engine.say(word)  
    engine.runAndWait() 

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            preprocessed_image = preprocess_image(frame)
            preprocessed_landmarks = preprocess_landmarks(hand_landmarks)

            prediction = model.predict([preprocessed_image, preprocessed_landmarks])
            predicted_letter = chr(np.argmax(prediction) + ord('A'))  

            if predicted_letter == previous_prediction:
            
                if start_time and (time.time() - start_time) >= 4:
                    word += predicted_letter  
                    print(f"Character '{predicted_letter}' added to the word.")
                    start_time = None  
            else:
                previous_prediction = predicted_letter
                start_time = time.time()  

            cv2.putText(frame, f'Predicted: {predicted_letter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f'Word: {word}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        print(f"Speaking the word: {word}")
        speak_word(word)  

    elif key == ord(' '):
        word += ' ' 
        print("Space added to the word.")

    elif key == 8:  
        if len(word) > 0:
            word = word[:-1]  
            print("Last character removed.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
