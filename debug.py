# Copyright (c) CUBOX, Inc. and its affiliates.
import cv2
import dlib
import numpy as np
import os

def visualize_dlib_landmarks(image_path, predictor_path="data/shape_predictor_68_face_landmarks.dat"):
    """
    Visualize landmarks detected by Dlib on a given image
    
    Args:
        image_path (str): Path to the image file
        predictor_path (str): Path to Dlib shape predictor file
    """
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray, 1)
    
    print(f"Detected {len(faces)} faces in {image_path}")
    
    # Draw rectangles around detected faces
    for rect in faces:
        # Draw face rectangle
        cv2.rectangle(img, 
                      (rect.left(), rect.top()), 
                      (rect.right(), rect.bottom()), 
                      (0, 255, 0), 
                      2)
        
        # Detect landmarks
        shape = predictor(gray, rect)
        
        # Draw landmarks
        for i in range(shape.num_parts):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            
            # Optionally, draw landmark numbers
            cv2.putText(img, str(i), (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Resize image if too large
    max_width = 1200
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Show the image
    cv2.imshow("Dlib Landmark Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batch_visualize_landmarks(directory_path="images"):
    """
    Batch process and visualize landmarks for all images in a directory
    
    Args:
        directory_path (str): Path to directory containing images
    """
    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    
    # Iterate through images in the directory
    for filename in sorted(os.listdir(directory_path)):
        # Check if file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            full_path = os.path.join(directory_path, filename)
            
            print(f"\nProcessing image: {filename}")
            try:
                visualize_dlib_landmarks(full_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Main execution
if __name__ == "__main__":
    visualize_dlib_landmarks("images/02.jpg")
