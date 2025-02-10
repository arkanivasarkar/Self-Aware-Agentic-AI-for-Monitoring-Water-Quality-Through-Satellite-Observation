import os
import numpy as np
import datetime as dt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths to data
base_folder = ".\\sentinel_satellite_images"
locations = [f"location{i}" for i in range(1, 11)]  # locations of 10 fish farms

# Function to preprocess image and extract features
def extract_image_features(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array


def extract_features_from_images(image_folder):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Prepare image data
    image_features = {}
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            date_str = filename.split(".png")[0]  # Extract date from filename
            date = dt.datetime.strptime(date_str, "%Y-%m-%d")
            image_path = os.path.join(image_folder, filename)
            img_array = extract_image_features(image_path)
            features = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
            image_features[date] = features
    return image_features

# Calculate the mean intensity of an image
def calculate_mean_intensity(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    mean_intensity = np.mean(img_array)  # Mean intensity of the image
    return mean_intensity

# Process data for all locations
def process_data():
    all_features = {"algal_bloom": [], "chlorophyll": [], "dissolved_oxygen": []}
    all_targets = {"algal_bloom": [], "chlorophyll": [], "dissolved_oxygen": []}

    for location in locations:
        location_path = os.path.join(base_folder, location)

        for target in ["algal_bloom", "chlorophyll", "dissolved_oxygen"]:
            print(f"  Extracting features for {target}...")
            image_folder = os.path.join(location_path, target)
            image_features = extract_features_from_images(image_folder)

            # Create sequential data from images (using image dates as time-series)
            dates = sorted(image_features.keys())
            features = [image_features[date] for date in dates]
            
             # Calculate target (mean intensity for each image)
            targets = [calculate_mean_intensity(os.path.join(image_folder, f"{date}.png")) for date in dates]

            all_features[target].append(np.array(features))
            all_targets[target].append(targets)

    for key in all_features.keys():
        all_features[key] = np.vstack(all_features[key])
        all_targets[key] = np.hstack(all_targets[key])

    return all_features, all_targets


# Train and save LSTM models
def train_and_save_timeseries_models(features, targets):
    models = {}
    for target in features.keys():
        X = features[target]
        y = targets[target]

        # Create time-series sequences
        sequence_length = 30  # Use 30-day windows
        X_sequences = []
        y_sequences = []
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

        # Define LSTM model
        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)  # Single output for regression
        ])
        model.compile(optimizer="adam", loss="mse")

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

        # Save the model
        model.save(f"{target}_lstm_model.h5")

        models[target] = model
    return models



if __name__ == "__main__":
    features, targets = process_data()
    train_and_save_timeseries_models(features, targets)
