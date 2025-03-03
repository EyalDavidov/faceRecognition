import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the frame height

counter = 0  # Initialize frame counter

# Initialize face_match flag
face_match = False  # Flag to indicate if face matches
valueErr = False  # Flag to indicate if there is a value error
distance = 0  # Initialize distance variable

faces = ['buzz.jpg', 'goofy.jpg', 'maui.webp', 'moana.jpg', 'potato.webp', 'shrek.jpg']  # List of reference face images
map = {}  # Dictionary to store distances for each reference image
current_face = 0  # Index of the current reference face
refImg = faces[current_face]  # Current reference image

def check_face(frame):
    global face_match
    global valueErr
    global distance
    global refImg
    refImg = faces[current_face]  # Update reference image

    reference_img = cv2.imread(refImg)  # Read the reference image
    if reference_img is None:
        print(f"Error: Could not load reference image {refImg}")
        return

    try:
        print(f"Verifying face with reference image: {refImg}")
        result = DeepFace.verify(frame, reference_img.copy(), model_name="SFace", distance_metric="cosine", detector_backend="yolov8")
        
        distance = result["distance"]  # Get the distance from the result
        if distance:
            valueErr = False
            if refImg not in map:
                map[refImg] = distance  # Store the distance if not already in map
            elif distance < map[refImg]:
                map[refImg] = distance  # Update the distance if the new one is smaller
        
    except ValueError as e:
        valueErr = True
        print(f"Value Error: {e}")

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()  # Start a new thread to check the face
            except ValueError:
                pass
        counter += 1  # Increment the counter
        if counter % 60 == 0:
            current_face += 1  # Move to the next reference face
            if current_face >= len(faces):
                print(map)  # Print the map of distances
                break
        face_img = cv2.imread(faces[current_face])  # Read the current reference face image
        face_img = cv2.resize(face_img, (100, 100))  # Resize the face image to fit in the frame

        frame[0:100, 0:100] = face_img  # Place the face image at the top-left corner of the frame

        if valueErr:
            cv2.putText(frame, "VALUE ERROR", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display value error message
        else:
            cv2.putText(frame, str(distance), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display the distance
        
        cv2.imshow("video", frame)  # Show the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == ord("q"):
        break  # Exit the loop if 'q' is pressed

cv2.destroyAllWindows()  # Close all OpenCV windows

cv2.destroyAllWindows()