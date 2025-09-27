import cv2
import numpy as np
import time

# -----------------------------
# Setup Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)  # Works cross-platform
time.sleep(2)  
background = 0

print("Preparing camera... Stay still for a few seconds.")
# Capture background frame (average of 60 frames for stability)
for i in range(60):
    ret, frame = cap.read()
    if not ret:
        continue
    frame = np.flip(frame, axis=1)  # horizontal flip
    background = frame if i == 0 else cv2.addWeighted(background, i/(i+1), frame, 1/(i+1), 0)

print("Background captured ✅")

# Create a resizable window
cv2.namedWindow("🪄 Invisibility Cloak", cv2.WINDOW_NORMAL)

# -----------------------------
# Setup Video Writer
# -----------------------------
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20  # You can adjust this
output_file = "invisibility_cloak_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# -----------------------------
# Start processing video
# -----------------------------
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = np.flip(img, axis=1)  # horizontal flip

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color masks
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 + mask2

    # Morphology to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    mask_inv = cv2.bitwise_not(mask)

    # Extract regions
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(img, img, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show result
    cv2.imshow("🪄 Invisibility Cloak", final_output)

    # Write the frame to the video file
    out.write(final_output)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as {output_file} ✅")
