import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
digits = datasets.load_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train a KNN classifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Function to process the input image
def process_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the classifier
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    # Flatten the image
    flattened = resized.flatten()
    # Scale the values to range [0, 16] to match the dataset
    scaled = flattened / 16.0
    return scaled

# Load the input image
input_img = cv2.imread('sign.jpg')

# Process the input image
processed_img = process_image(input_img)

# Predict the sign
prediction = clf.predict([processed_img])

# Print the predicted sign
print("Predicted Sign:", prediction[0])
