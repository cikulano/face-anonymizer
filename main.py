import cv2
import mediapipe as mp
import os
import argparse

def process_img(image,face_detection):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    out = face_detection.process(image_rgb)  # Process the image with face detection

    if out.detections is not None:  # If there are detections
        for detection in out.detections:  # Iterate over each detection
            location_data = detection.location_data  # Get the location data of the detection
            bounding_box = location_data.relative_bounding_box  # Get the relative bounding box of the detection

            # Calculate the coordinates of the bounding box
            x = int(bounding_box.xmin * image.shape[1])
            y = int(bounding_box.ymin * image.shape[0])
            w = int(bounding_box.width * image.shape[1])
            h = int(bounding_box.height * image.shape[0])

            # Blur the image according to the bounding box
            blurred_image = image[y:y+h, x:x+w]
            blurred_image = cv2.blur(blurred_image, (30, 30))

            # Replace the blurred region with the original image
            image[y:y+h, x:x+w] = blurred_image
    return image

output_dir = "./output_dir"  # Path to the output directory
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Create an argument parser
args = argparse.ArgumentParser()

# Add arguments to the parser
args.add_argument("--mode", default='webcam')  # Specify the mode (default: 'image')
args.add_argument("--FilePath", default=None)  # Specify the file path (default: './data/self.jpg')

args = args.parse_args()

# Detect faces
mp_face_detection = mp.solutions.face_detection  # Create a face detection object
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":  # If the mode is 'image'
        image = cv2.imread(args.FilePath)  # Read the image
        process_img(image, face_detection)  # Call the function process_img
        output_path = os.path.join(output_dir, "blurred_image.jpg")  # Path to save the blurred image
        cv2.imwrite(output_path, image)  # Save the blurred image to the output directory

    elif args.mode == "video":  # If the mode is 'video'
        cap = cv2.VideoCapture(args.FilePath)  # Open the specified file path
        ret, frame = cap.read() 
        # Create a VideoWriter object
        video_output = cv2.VideoWriter(os.path.join(output_dir,'output.mp4'),
                                        cv2.VideoWriter_fourcc(*'MP4V'),
                                        25, 
                                        (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)  # Call the function process_img and assign the processed frame back to 'frame'

            video_output.write(frame)

            ret, frame = cap.read()  # Read a frame from the camera
        
        cap.release()
        video_output.release()

    elif args.mode == "webcam":  # If the mode is 'webcam'
        cap = cv2.VideoCapture(0)  # Open the webcam
        while True:
            ret, frame = cap.read()  # Read a frame from the webcam
            frame = process_img(frame, face_detection)  # Call the function process_img and assign the processed frame back to 'frame'
            cv2.imshow('Blurred Webcam', frame)  # Display the blurred frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # If 'q' is pressed, break the loop
                break
        cap.release()
        cv2.destroyAllWindows()
    




