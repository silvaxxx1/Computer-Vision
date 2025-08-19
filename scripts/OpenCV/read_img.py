# Task : Read and display an image


import cv2
import sys

# Path to the image
img_path = '/home/silva/SILVA.AI/Projects/Computer Vision/assets/fero.jpg'

# Load the image
img = cv2.imread(img_path,0)

# Check if image is loaded
if img is None:
    print(f"Error: Image not found at {img_path}")
    sys.exit(1)

print("Image loaded successfully.")
print("Image shape:", img.shape)

# Display the image in a window
cv2.imshow('Fero Image', img)

# Wait for a key press:
# ESC key (27) -> exit without saving
# 's' key -> save image and exit
k = cv2.waitKey(0)
if k == 27:  # ESC
    cv2.destroyAllWindows()
elif k == ord('s'):
    save_path = '/home/silva/SILVA.AI/Projects/Computer Vision/assets/fero_saved.jpg'
    cv2.imwrite(save_path, img)
    print(f"Image saved to {save_path}")
    cv2.destroyAllWindows() 

