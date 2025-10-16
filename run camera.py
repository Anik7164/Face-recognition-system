import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

class RealTimeEmotionDetection:
    def __init__(self, model_path):
        # Load your trained model
        try:
            self.model = load_model(model_path)
            print(">>> Model loaded successfully!")
        except Exception as e:
            print(f">>> Error loading model: {e}")
            return
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # For smoothing predictions
        self.prediction_history = deque(maxlen=15)
        
        # Default emotions (update these after checking your model)
        self.emotions = {
            0: "Angry",
            1: "Disgust", 
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }
        
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
    def detect_emotions_from_camera(self):
        """Detect emotions in real-time from webcam"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print(">>> Error: Could not open webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(">>> Starting real-time emotion detection...")
        print(">>> Press 'q' to quit")
        print(">>> Press 'r' to reset emotion history")
        print(">>> Press 's' to save current frame")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(">>> Error: Could not read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100),  # Increased minimum size for better quality
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Extract and preprocess face
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to match model input (48x48)
                    face_resized = cv2.resize(face_roi, (48, 48))
                    
                    # Normalize and reshape for model
                    face_normalized = face_resized.astype('float32') / 255.0
                    face_input = face_normalized.reshape(1, 48, 48, 1)
                    
                    # Predict emotion
                    predictions = self.model.predict(face_input, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    # Add to history for smoothing
                    self.prediction_history.append(emotion_idx)
                    
                    # Get smoothed emotion (most frequent in last 15 frames)
                    if len(self.prediction_history) >= 5:
                        emotion_counts = np.bincount(self.prediction_history)
                        smoothed_emotion = np.argmax(emotion_counts)
                        current_emotion = smoothed_emotion
                    else:
                        current_emotion = emotion_idx
                    
                    # Draw face rectangle
                    color = self.get_emotion_color(current_emotion)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw emotion text with confidence
                    emotion_text = f"{self.emotions.get(current_emotion, 'Unknown')}: {confidence:.2f}"
                    cv2.putText(frame, emotion_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Draw confidence bar
                    self.draw_confidence_bar(frame, x, y+h+10, w, confidence, color)
                    
                    # Draw all emotions probabilities (on the side)
                    self.draw_emotion_probabilities(frame, predictions[0], x+w+10, y)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Display frame
            cv2.imshow('Real-Time Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.prediction_history.clear()
                print(">>> Emotion history reset")
            elif key == ord('s'):
                cv2.imwrite('emotion_capture.jpg', frame)
                print(">>> Frame saved as 'emotion_capture.jpg'")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(">>> Emotion detection stopped")
    
    def get_emotion_color(self, emotion_idx):
        """Get color based on emotion"""
        colors = {
            0: (0, 0, 255),    # Angry - Red
            1: (0, 128, 0),    # Disgust - Green
            2: (128, 0, 128),  # Fear - Purple
            3: (0, 255, 255),  # Happy - Yellow
            4: (255, 255, 255), # Neutral - White
            5: (255, 0, 0),    # Sad - Blue
            6: (0, 165, 255)   # Surprise - Orange
        }
        return colors.get(emotion_idx, (255, 255, 255))
    
    def draw_confidence_bar(self, frame, x, y, width, confidence, color):
        """Draw confidence bar below face"""
        bar_height = 15
        fill_width = int(width * confidence)
        
        # Background
        cv2.rectangle(frame, (x, y), (x+width, y+bar_height), (50, 50, 50), -1)
        # Fill
        cv2.rectangle(frame, (x, y), (x+fill_width, y+bar_height), color, -1)
        # Border
        cv2.rectangle(frame, (x, y), (x+width, y+bar_height), color, 1)
    
    def draw_emotion_probabilities(self, frame, probabilities, start_x, start_y):
        """Draw all emotion probabilities on the side"""
        bar_width = 120
        bar_height = 15
        spacing = 5
        
        for i, (emotion_name, prob) in enumerate(zip(self.emotions.values(), probabilities)):
            y_pos = start_y + i * (bar_height + spacing)
            
            # Background
            cv2.rectangle(frame, 
                         (start_x, y_pos),
                         (start_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Fill
            fill_width = int(bar_width * prob)
            color = self.get_emotion_color(i)
            cv2.rectangle(frame, 
                         (start_x, y_pos),
                         (start_x + fill_width, y_pos + bar_height),
                         color, -1)
            
            # Text
            text = f"{emotion_name}: {prob:.2f}"
            cv2.putText(frame, text, 
                       (start_x + bar_width + 5, y_pos + bar_height - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    # Initialize emotion detection with your model
    model_path = 'facial_expression_model.h5'  # Your trained model
    
    detector = RealTimeEmotionDetection(model_path)
    
    # Start real-time detection
    detector.detect_emotions_from_camera()

if __name__ == "__main__":
    main()