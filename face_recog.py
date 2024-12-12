import math
import os
import cv2
import face_recognition
import numpy as np
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (2.0 * range)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceReco:
    def __init__(self):
        self.face_locations = []
        self.known_face_encodings = []
        self.face_names = []
        self.known_face_names = []
        self.process_frame = True

        print("Initializing face recognition...")
        self.encode('faces')  # Adjust the folder name as necessary

    def encode(self, folder):
    # Check if folder exists
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            return

        for image in os.listdir(folder):
            if image.endswith(('.png', '.jpg', '.jpeg')):
                # Load the image using face_recognition
                img_path = os.path.join(folder, image)
                f_image = face_recognition.load_image_file(img_path)

                # Check if the image is in the correct format
                if f_image.ndim != 3 or f_image.shape[2] != 3:
                    print(f"Image {image} is not a valid RGB image.")
                    continue

                # Get the face encodings
                encodings = face_recognition.face_encodings(f_image)

                if encodings:
                    f_encode = encodings[0]
                    self.known_face_encodings.append(f_encode)
                    # Use the file name without the extension
                    self.known_face_names.append(os.path.splitext(image)[0])
                    print(f"Encoded {image}")
                else:
                    print(f"No face found in {image}")

        print(f"Encoding complete. Known faces: {self.known_face_names}")


    def run_reco(self):
        # Start webcam video capture
        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
            rgb_frame = frame[:, :, ::-1]

            # Find all face locations and encodings in the current frame
            self.face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

            # Initialize an array for face names
            self.face_names = []

            # Loop through the face encodings to recognize faces
            for face_encoding in face_encodings:
                # Compare the encodings to the known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

            # Display the results on the frame
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with the name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facep = FaceReco()
    facep.run_reco()  # Start face recognition using the webcam
