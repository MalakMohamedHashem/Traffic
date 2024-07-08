import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Get the list of category directories
    categories = os.listdir(data_dir)
    categories.sort()  # Ensure categories are sorted numerically

    for category in categories:
        category_path = os.path.join(data_dir, category)

        if not os.path.isdir(category_path):
            continue  # Skip if not a directory

        # Get the label (category) for the current directory
        label = int(category)

        # Load images from the current category directory
        image_files = os.listdir(category_path)
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)

            # Read the image using OpenCV (assuming it's RGB format)
            image = cv2.imread(r"C:\Users\pc\Desktop\traffic\gtsrb\gtsrb\0\00000_00000.ppm")
            if image is None:
                continue  # Skip if unable to read the image

            # Resize the image to the specified dimensions (IMG_WIDTH x IMG_HEIGHT)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Append the image and label to the lists
            images.append(image)
            labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels



def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Flatten layer to transition from convolutional to fully connected layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_categories, activation='softmax'))  # Output layer with softmax activation

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
                  metrics=['accuracy'])

    return model


# Define constants
IMG_WIDTH = 100
IMG_HEIGHT = 100
NUM_CATEGORIES = 10  # Example: Number of output categories (classes)

# Create and compile the CNN model
input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)  # Assuming RGB images (3 channels)
cnn_model = create_cnn_model(input_shape, NUM_CATEGORIES)

# Display the model summary
cnn_model.summary()


if __name__ == "__main__":
    main()
