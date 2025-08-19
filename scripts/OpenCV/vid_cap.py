# Task : capture video from camera
import cv2 

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
# fourcc to encode video quality
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'cap.avi', # name of output file
    fourcc, # codec
    20.0, # frames per second
    (640, 480) # width and height 
) 

cap.set(3, 500)
cap.set(4, 500)


while cap.isOpened(): # check if the video is opened
    ret, frame = cap.read() # read the frame
    if ret: # check if the frame is read
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame) # display the frame
        out.write(frame) # write the frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
            break
    else:
        break

cap.release() # release the video capture object
out.release() # release the video writer object
cv2.destroyAllWindows() # close all windows


