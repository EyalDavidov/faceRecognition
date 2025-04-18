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

faces = [ 'happy', 'surprised']  # List of reference face images
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
            map[refImg] = distance 
        
    except ValueError as e:
        valueErr = True
        print(f"Value Error: {e}")


time.sleep(1)  # Sleep the main thread for 2 seconds

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    if ret:
        if counter <= 60:
            if counter < 20:
                cv2.putText(frame, "3", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            elif 20 < counter <= 40:
                cv2.putText(frame, "2", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            elif 40 < counter <= 60:
                cv2.putText(frame, "1", (300, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
        elif counter > 60:
            if counter % 40 == 0 :
                try:
                    print("sampling face...")
                    threading.Thread(target=check_face, args=(frame.copy(), current_face)).start()
                    # threading.Thread(target=check_face, args=(frame.copy(),)).start()  # Start a new thread to check the face
                    print("switching face...")
                    current_face += 1  # Move to the next reference face
                    if current_face > len(faces):
                        break
                except ValueError:
                    pass
                
            face_img = cv2.imread(refImgPath)  # Read the current reference face image
            x, y, _ = face_img.shape  # Unpack height, width, and ignore channels
            frame[0:x, 0:y] = face_img  # Place the face image at the top-left corner of the frame

            if valueErr:
                cv2.putText(frame, "VALUE ERROR", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display value error message
            else:
                cv2.putText(frame, str(distance), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Display the distance
                print(f"Distance: {distance}")  # Print the distance to the console

        counter += 1  # Increment the counter
        cv2.imshow("video", frame)  # Show the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == ord("q"):
        break  # Exit the loop if 'q' is pressed

# Wait for all child threads to finish
for thread in threading.enumerate():
    if thread is not threading.main_thread():
        thread.join()
print(map)  # Print the map of distances

# Calculate and print final grades based on the average of the map
if map:
    avg_distance = sum(map.values()) / len(map)
    print(f"Final Grade (Average Distance): {avg_distance:.2f}")
else:
    print("No distances were calculated.")

cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()