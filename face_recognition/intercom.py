from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import json
import cv2
import time
import os
import face_recognition
import photo_and_encode_auto as ph_en
import facial_req2 as fr
import numpy as np
import sys
import uuid
from imutils import paths
import pickle
import shutil
import subprocess
from datetime import datetime



vid = ""
people_list = []

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

def generate_frames():
    stream_from = 0
    if os.path.exists("./camera_data.json"):
    # Open and read the JSON file
        with open("./camera_data.json", 'r') as json_file:
            data = json.load(json_file)
            
            # Check if all the necessary information is available
            if 'camera_ip' in data and 'camera_name' in data and 'camera_password' in data:
                camera_ip = data['camera_ip']
                camera_name = data['camera_name']
                camera_password = data['camera_password']
                
                stream_from = f"rtsp://{camera_name}:{camera_password}@{camera_ip}:554/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(stream_from)
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > 15:
            return
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 

def detect_person(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return face coordinates if at least one face is detected, otherwise None
    if len(faces) > 0:
        return faces[0]  # Return coordinates of the first detected face
    else:
        return None

# Main function to capture video from IP camera
def showVid():
    stream_from = 0
    if os.path.exists("./camera_data.json"):
    # Open and read the JSON file
        with open("./camera_data.json", 'r') as json_file:
            data = json.load(json_file)
            
            # Check if all the necessary information is available
            if 'camera_ip' in data and 'camera_name' in data and 'camera_password' in data:
                camera_ip = data['camera_ip']
                camera_name = data['camera_name']
                camera_password = data['camera_password']
                
                # Create the stream_from string
                stream_from = f"rtsp://{camera_name}:{camera_password}@{camera_ip}:554/cam/realmonitor?channel=1&subtype=1"
    return  fr.recognize_faces(stream_from)


@app.route('/')
def index():
    return render_template('home.html', vid=vid)

@app.route('/outside')
def outside():
    return render_template('outside.html')

server_state = {
    'action_needed': False
}

@app.route('/set_flag', methods=['POST'])
def set_flag():
    server_state['action_needed'] = True
    return jsonify({'success': True})

@app.route('/check_flag', methods=['GET'])
def check_flag():
    action_needed = server_state['action_needed']
    if action_needed:
        server_state['action_needed'] = False  # Reset the flag after checking
    return jsonify({'action_needed': action_needed})

@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Get the form data
        camera_ip = request.form['camera_ip']
        camera_name = request.form['camera_name']
        camera_password = request.form['camera_password']
        app_password = request.form['app_password']

        # Dictionary that holds the data
        data = {
            'camera_ip': camera_ip,
            'camera_name': camera_name,
            'camera_password': camera_password,
            'app_password': app_password
        }

        # data -> JSON
        with open('camera_data.json', 'w') as json_file:
            json.dump(data, json_file)

        # Print success message
        return redirect(url_for('saved_data'))

    return render_template('settings.html')

@app.route('/pw')
def pw():
    with open("./camera_data.json", 'r') as json_file:
        data = json.load(json_file)        
        # Check if all the necessary information is available
        if 'app_password' in data:
            app_pass = data['app_password']
            return app_pass
        else:
            return ''


@app.route('/saved_data')
def saved_data():
    return 'Data saved successfully!'

@app.route('/alarm', methods=['GET', 'POST'])
def alarm():
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d at %H:%M:%S")
    phone_number = "1234567890"  # Replace with the target phone number
    message = "Alarm set off: " + formatted_now + "!"
    script = 'send_imessage.scpt'
    subprocess.run(['osascript', script, phone_number, message])
    return jsonify(success=True)
    
@app.route('/unlock')
def unlock():
    print("unlocked")
    return jsonify(success=True)

@app.route('/accounts')
def accounts():
    folder_path = './static/known_people/'
    folder_items = os.listdir(folder_path)
    return render_template('accounts.html', folder_items=folder_items)

@app.route('/function2')
def function2():
    name = showVid()
    return name

@app.route('/people')
def people():
    folder_dir = "./static/known_people/"
    return jsonify(os.listdir(folder_dir))

@app.route('/rename-image', methods=['POST'])
def rename_image():
    data = request.get_json()
    old_name = data.get('oldName')
    new_name = data.get('newName')
    folderold = old_name[:4]
    foldernew = new_name[:4]
    os.rename("./static/known_people/"+old_name, "./static/known_people/"+new_name)
    os.rename("./static/dataset/"+folderold, "./static/dataset/"+foldernew)
    ph_en.encode()
    return jsonify({'message': 'Image renamed successfully'})

@app.route('/delete-image', methods=['POST'])
def delete_image():
    data = request.get_json()
    image_name = data.get('imageName')
    folder = image_name[:-4]
    os.remove("./static/known_people/"+image_name)
    shutil.rmtree("./static/dataset/"+folder)
    ph_en.encode()

@app.route('/add_person', methods=['POST'])
def add_person():
    data = request.get_json()
    name = data.get('name')    
    ph_en.take_photos(name)
    return Response(ph_en.encode(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
