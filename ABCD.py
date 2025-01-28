import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def show_image(image, title="Image"):
    # Use matplotlib to display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')  # Hide the axes
    plt.show()

def extract_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Original Image
    cv2.imwrite('output_original.jpg', image)  # Save original image
    show_image(image, 'Original Image')

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output_grayscale.jpg', gray)  # Save grayscale image
    show_image(gray, 'Grayscale Image')

    # Step 3: Use Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('output_blurred.jpg', blurred)  # Save blurred image
    show_image(blurred, 'Blurred Image')

    # Step 4: Use edge detection (Canny)
    edges = cv2.Canny(blurred, 100, 200)
    cv2.imwrite('output_edges.jpg', edges)  # Save edge-detected image
    show_image(edges, 'Edges Image')

    # Step 5: Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Loop through contours and find rectangles
    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check if the contour has four points (a rectangle)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y + h, x:x + w]  # Region of interest (the potential number plate)

            # Highlight the detected region on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imwrite('output_detected_number_plate.jpg', image)  # Save detected number plate image
            show_image(image, 'Detected Number Plate')

            # Step 7: Use Tesseract to extract text from the region of interest
            text = pytesseract.image_to_string(roi, config='--psm 8')
            print("Extracted Number Plate:", text.strip())

            # Save the ROI (Number Plate region) for further inspection
            cv2.imwrite('output_roi.jpg', roi)
            show_image(roi, 'Region of Interest')

# Example usage
extract_number_plate('car1.jpg')
