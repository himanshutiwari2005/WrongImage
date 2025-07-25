import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras._tf_keras.keras.losses import MeanSquaredError

IMG_SIZE = 128  # Resize all images to 128x128
THRESHOLD_FACTOR = 0.003 # How strict anomaly detection is

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            filenames.append(filename)
    return np.array(images)[..., np.newaxis], filenames

def build_autoencoder():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))

def main(train_dir, test_dir, ground_truth_labels=None, reuse_model=False, model_path="autoencoder_model.h5"):
    print("Loading data...")
    x_train, _ = load_images_from_folder(train_dir)
    x_test, test_filenames = load_images_from_folder(test_dir)

    if reuse_model and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        autoencoder = tf.keras.models.load_model(model_path)
    else:
        print("Building and training autoencoder...")
        autoencoder = build_autoencoder()
        autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
        autoencoder.fit(x_train, x_train, epochs=50, batch_size=16, verbose=1)
        autoencoder.save(model_path)
        print(f"Model saved to {model_path}")

    print("Computing reconstruction error...")
    reconstructions = autoencoder.predict(x_test)
    errors = calculate_reconstruction_error(x_test, reconstructions)

    # Threshold based on training set (even if model is loaded, this step ensures correct threshold)
    train_reconstructions = autoencoder.predict(x_train)
    train_errors = calculate_reconstruction_error(x_train, train_reconstructions)
    threshold = np.mean(train_errors) + THRESHOLD_FACTOR * np.std(train_errors)

    anomaly_flags = errors > threshold

    print("\n=== Anomaly Detection Results ===")
    for fname, err, flag in zip(test_filenames, errors, anomaly_flags):
        print(f"{fname}: {'Anomaly' if flag else 'Normal'} (Error: {err:.6f})")

    if ground_truth_labels is not None:
        precision = precision_score(ground_truth_labels, anomaly_flags)
        recall = recall_score(ground_truth_labels, anomaly_flags)
        f1 = f1_score(ground_truth_labels, anomaly_flags)
        print(f"\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")


if __name__ == "__main__":
    train_folder = "data/metal_nut/metal_nut/train/good"
    test_folder = "data/metal_nut/metal_nut/test/good"
    ground_truth = None  # Replace with actual if known

    # Set reuse_model=True to skip retraining
    main(train_folder, test_folder, ground_truth_labels=ground_truth, reuse_model=True)
