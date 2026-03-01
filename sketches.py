import cv2
from utils import sketch

cap = cv2.VideoCapture(0) #it still uses my phone
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    sketched = sketch(frame)
    cv2.imshow("Sketch", sketched)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
