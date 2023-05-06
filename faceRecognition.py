

# Reference Link 
#https://www.datacamp.com/tutorial/face-detection-python-opencv

import cv2
import matplotlib.pyplot as plt

imagePath = 'tony.jpg'


# Read the Image
img = cv2.imread(imagePath)

# check the shaper of array 
print(img.shape)

# Convert the Image to Grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



print(gray_image.shape)

# Load the Classifier

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img_rgb)

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# cv2.imshow(img_rgb)