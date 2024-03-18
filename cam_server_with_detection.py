from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model

import os

app = Flask(__name__)

# Set up a global variable to store the latest frame
global_frame = None

# Define the path to the model file
model_file = "/home/januka_pi/Documents/GitHub/AgroMate/keras_model.h5"

# Load the Keras model
model = load_model(model_file, compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def detect_objects(frame):
    # Resize the image
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Overlay class name and confidence score on the image
    text = f"Class: {class_name[2:]} | Confidence: {str(np.round(confidence_score * 100))[:-2]}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

def gen():
    while True:
        if global_frame is not None:
            # Perform object detection on the frame
            detected_frame = detect_objects(global_frame)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', detected_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    global global_frame
    frame_data = request.data
    frame_np = np.frombuffer(frame_data, dtype=np.uint8)
    global_frame = cv2.imdecode(frame_np, flags=1)  # 1 means load color image
    return 'OK'

@app.route('/preview')
def preview():
    return render_template('preview.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
