import cv2
import numpy as np
import face_recognition
import os

path = '/Users/ishankanodia/Desktop/SEM 5/Innovation Lab/fold'    # Path to the folder containing images of the persons
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load images and class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find encodings of faces in the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure there is at least one encoding
            encodeList.append(encode[0])
    return encodeList

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# List to store recognized names
recognized_names = []

# Load the image to process (replace the webcam feed with the PNG image)
img_path = '/Users/ishankanodia/Downloads/img.png'  # Use the uploaded image path
img = cv2.imread(img_path)

if img is None:
    print("Error loading image.")
else:
    # Convert color for face recognition processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current image
    facesCurFrame = face_recognition.face_locations(imgRGB)
    encodesCurFrame = face_recognition.face_encodings(imgRGB, facesCurFrame)

    # Compare detected faces with known faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "UNKNOWN"

        # Add recognized name to the list
        recognized_names.append(name)

        # Draw rectangle around the face
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the image with face recognition annotations
    cv2.imshow('Image', img)
    cv2.waitKey(0)  # Wait until a key is pressed to close the window

cv2.destroyAllWindows()
