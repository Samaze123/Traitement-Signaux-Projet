import cv2

# Load the image
img_white_rectangle = cv2.imread(
    "./tests/white_rectangle.png"
)  # Replace 'your_image.jpg' with your image file path

# Convert the image to grayscale
gray = cv2.cvtColor(img_white_rectangle, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate white from black
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming there is only one white rectangle, find its bounding box
if len(contours) > 0:
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image to the bounding box
    cropped_image = img_white_rectangle[y : y + h, x : x + w]

    # Display the image
    cv2.imshow("Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display or further process the cropped image as needed
else:
    print("No white rectangle found in the image.")

# Release any OpenCV resources (not always necessary, but good practice)
cv2.destroyAllWindows()
