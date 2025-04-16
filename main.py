import threading

import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the frame height

counter = 1  # Initialize frame counter

# Initialize face_match flag
face_match = False  # Flag to indicate if face matches
valueErr = False  # Flag to indicate if there is a value error
distance = 0  # Initialize distance variable

faces = ['angry', 'discomfort', 'disgusted', 'frightened', 'happy', 'surprised']  # List of reference face images
map = {}  # Dictionary to store distances for each reference image
current_face = 0  # Index of the current reference face
refImg = faces[current_face]  # Current reference image
refImgPath = f"images/{refImg}.png"

def check_face(frame, current_face):
    global face_match
    global valueErr
    global distance
    global refImg
    global refImgPath
    global map
    refImg = faces[current_face]  # Update reference image
    refImgPath = f"images/{refImg}.png"


    reference_img = cv2.imread(refImgPath)  # Read the reference image
    if reference_img is None:
        print(f"Error: Could not load reference image {refImg}")
        return

    try:
        print(f"Verifying face with reference image: {refImg}")
        result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False,silent=True)  # Verify the face using DeepFace
        
        distance = round(100 - result["distance"] * 100, 2)  # Get the distance from the result with 2 decimal points
        if distance:
            valueErr = False
            if refImg not in map:
                map[refImg] = distance  # Store the distance if not already in map
            elif distance < map[refImg]:
                map[refImg] = distance  # Update the distance if the new one is smaller
        
    except ValueError as e:
        valueErr = True
        print(f"Value Error: {e}")


time.sleep(1)  # Sleep the main thread for 2 seconds


while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if ret:
        if counter % 90 == 0 :
            try:
                print("---------------------------------------------------------")
                print("sampling face...")
                print("---------------------------------------------------------")
                threading.Thread(target=check_face, args=(frame.copy(), current_face)).start()
                # threading.Thread(target=check_face, args=(frame.copy(),)).start()  # Start a new thread to check the face
                print("---------------------------------------------------------")
                print("switching face...")
                print("---------------------------------------------------------")
                current_face += 1  # Move to the next reference face
                if current_face >= len(faces):
                    break
            except ValueError:
                pass

        counter += 1  # Increment the counter
            
        face_img = cv2.imread(refImgPath)  # Read the current reference face image
        face_img = cv2.resize(face_img, (200, 200))  # Resize the face image to fit in the frame


        frame[0:200, 0:200] = face_img  # Place the face image at the top-left corner of the frame

        if valueErr:
            cv2.putText(frame, "VALUE ERROR", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display value error message
        else:
            cv2.putText(frame, str(distance), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display the distance
        
        cv2.imshow("video", frame)  # Show the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == ord("q"):
        break  # Exit the loop if 'q' is pressed

# Wait for all child threads to finish
for thread in threading.enumerate():
    if thread is not threading.main_thread():
        thread.join()
print(map)  # Print the map of distances

cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()