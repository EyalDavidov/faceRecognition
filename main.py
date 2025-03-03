import threading
#this is test 
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0

face_match = False
valueErr = False
reference_img = cv2.imread("ref.jpg")
distance = 0

def check_face(frame):
    global face_match
    global valueErr
    global distance

    try:
        result = DeepFace.verify(frame, reference_img.copy(), model_name="SFace", distance_metric="cosine",detector_backend="dlib")
        
        
        distance = result["distance"]
        if distance:
            valueErr = False
        
    except ValueError:
        valueErr = True


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target = check_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if valueErr:
            cv2.putText(frame,"VALUE ERROR",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,str(distance),(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        
        
        cv2.imshow("video",frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
