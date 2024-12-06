import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

# Load the trained model (pre-trained)
model = load_model('celebrity_model.h5')

# Load the saved label encoder
with open('labelencoder.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

def capture_and_predict():
    # Start the webcam for real-time predictions
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match the input size of the model
        img_resized = cv2.resize(frame, (250, 250))
        
        # Convert the image to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a numpy array and preprocess it for the model
        img_array = img_to_array(img_rgb) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        
        # Decode the prediction to the label
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = labelencoder.inverse_transform(predicted_class)

        # Display the resulting frame with prediction text
        cv2.putText(frame, f"Predicted: {predicted_label[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Camera", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_predict()
