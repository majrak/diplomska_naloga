import cv2
import sys
import uuid
import os
import time
from imutils import paths
import face_recognition
import pickle
import shutil

def take_photos(name):
    os.mkdir("./static/dataset/" + name)
    counter = 0

    cam = cv2.VideoCapture(0)

    while counter < 10:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        time.sleep(0.2)
        
        img_name = "static/dataset/{}/{}_{}.jpg".format(name, name, counter)
        
        # Write frame to file
        status = cv2.imwrite(img_name, frame)
        if counter == 1:
            shutil.copy("./static/dataset/"+name+"/"+name+"_1.jpg", "./static/known_people/"+name+".jpg")

        counter += 1

    # Release camera
    cam.release()
    cv2.destroyAllWindows()



def encode():
    print("[INFO] start processing faces...")
    imagePaths = list(paths.list_images("static/dataset"))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model="hog")

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()