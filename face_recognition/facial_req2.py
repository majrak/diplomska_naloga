from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

def recognize_faces(rtsp_url):
    currentname = "unknown"
    encodingsP = "encodings.pickle"
    cascade = "haarcascade_frontalface_default.xml"

    # load the known faces and embeddings along with OpenCV's Haar cascade for face detection
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())
    detector = cv2.CascadeClassifier(cascade)

    # initialize the video stream with RTSP URL
    print(f"[INFO] starting video stream from {rtsp_url}...")
    vs = VideoStream(src=rtsp_url).start()
    time.sleep(2.0)

    # start the FPS counter
    fps = FPS().start()

    # initialize variables for detecting time a face stays in the frame
    start_time = None
    detected_person = None

    # set end time for the program to run (10 seconds)
    end_time = time.time() + 10

    # loop over frames from the video stream
    while time.time() < end_time:
        # grab the frame from the threaded video stream and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        
        # convert the input frame from (1) BGR to grayscale (for face detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order but we need them in (top, right, bottom, left) order, so we need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown" # if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes
                name = max(counts, key=counts.get)
                
                if currentname != name:
                    currentname = name
                    start_time = time.time()

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)

            # Check if a person has stayed in the frame for 2 seconds
            if start_time is not None and time.time() - start_time > 2:
                detected_person = name
                break

        if detected_person is not None:
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    if detected_person is None:
        print("No known person detected")
        return "Unknown"
    else:
        return detected_person