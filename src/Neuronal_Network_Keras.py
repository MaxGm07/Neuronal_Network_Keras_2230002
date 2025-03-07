import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input   # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore

def E_Red_neuronal():
    """
    Function to create, train, and evaluate a neural network using the MNIST dataset.
    """
    # Load the MNIST dataset, which contains images of handwritten digits
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Display information about the training data
    print("Shape of training data:", train_data_x.shape)  # Size of training images (60000 images of 28x28 pixels)
    print("Label of the first training example:", train_labels_y[10])  # Label of the first image (which digit it is)
    print("Shape of test data:", test_data_x.shape)  # Size of test images (10000 images of 28x28 pixels)

    # Display one of the training images to visualize it
    plt.imshow(train_data_x[10])  # Show the image in grayscale
    plt.title("Example of a Training Image")  # Add a title to the image
    plt.show()  # Display the image in a window

    # Create the neural network model
    model = Sequential([
        Input(shape=(28 * 28,)),  # Input layer: images are flattened into a vector of 784 values (28x28)
        Dense(512, activation='relu'),  # Hidden layer: 512 neurons with ReLU activation (helps the network learn better)
        Dense(10, activation='softmax')  # Output layer: 10 neurons (one for each digit from 0 to 9) with softmax activation (to get probabilities)
    ])

    # Configure the model for training
    model.compile(
        optimizer='rmsprop',  # Use the RMSprop optimizer to adjust the weights of the network
        loss='categorical_crossentropy',  # Loss function: measures how poorly the network is learning
        metrics=['accuracy']  # Metric: we want to measure accuracy (percentage of correct predictions)
    )

    # Display a summary of the neural network architecture
    print("Model summary:")
    model.summary()  # Shows the number of layers, parameters, etc.

    # Prepare the training data for the network
    x_train = train_data_x.reshape(60000, 28 * 28)  # Flatten the 28x28 images into vectors of 784 values
    x_train = x_train.astype('float32') / 255  # Normalize pixel values to be between 0 and 1
    y_train = to_categorical(train_labels_y)  # Convert labels to one-hot encoding format

    # Prepare the test data in the same way
    x_test = test_data_x.reshape(10000, 28 * 28)  # Flatten the test images
    x_test = x_test.astype('float32') / 255  # Normalize pixel values
    y_test = to_categorical(test_labels_y)  # Convert test labels to one-hot encoding

    # Train the neural network
    print("Training the neural network...")
    model.fit(x_train, y_train, epochs=10, batch_size=128)  # Train for 10 epochs, using batches of 128 images

    # Evaluate the neural network on the test data
    print("Evaluating the neural network...")
    loss, accuracy = model.evaluate(x_test, y_test)  # Calculate loss and accuracy on the test set
    print(f"Loss: {loss}, Accuracy: {accuracy}")  # Display the results

    plt.show()  # Show any remaining plots