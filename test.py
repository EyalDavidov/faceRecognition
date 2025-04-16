import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    cv2.imshow("video", frame)  # Show the frame

    key = cv2.waitKey(1)  # Wait for a key press
    if key == ord("q"):
        break  # Exit the loop if 'q' is pressed

cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()  # Release the camera resource