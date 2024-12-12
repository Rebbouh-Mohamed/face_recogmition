import cv2

def capture_image():
    # Start video capture from the webcam (0 is the default camera)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture an image or 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the captured frame to RGB format (if needed)
        # OpenCV reads images in BGR format by default, we want RGB for saving
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the captured frame
        cv2.imshow('Webcam', frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF

        # If 'c' is pressed, save the image
        if key == ord('c'):
            image_name = "captured_image.jpg"  # Change the name if desired
            cv2.imwrite(image_name, frame)  # Save the captured frame
            print(f"Image saved as {image_name}")

        # If 'q' is pressed, exit the loop
        elif key == ord('q'):
            print("Quitting...")
            break

    # Release the capture and close any OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
