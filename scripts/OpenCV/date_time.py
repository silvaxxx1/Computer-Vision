import cv2 
import datetime 


cap = cv2.VideoCapture(0)

while cap.isOpened(): # check if the video is opened
    ret, frame = cap.read() # read the frame
    if ret: # check if the frame is read
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        text = 'width:' + str(cap.get(3)) + ' height:' + str(cap.get(4))
        date = str(datetime.datetime.now())
        frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 255), 2)
        frame = cv2.putText(frame, date, (10, 100), font, 1, (0, 255, 255), 2) 
        
        cv2.imshow('frame', frame) # display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
            break
    else:
        break

cap.release() # release the video capture object
cv2.destroyAllWindows() # close all windows 
