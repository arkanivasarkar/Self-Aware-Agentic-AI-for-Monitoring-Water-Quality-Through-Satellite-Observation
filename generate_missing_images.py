import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import os
from PIL import Image
from datetime import datetime, timedelta

#Load and Preprocess the Data
def load_image_data(data_dir):
    file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    dates = []

    for file_name in file_names:
        try:
            date_str = os.path.splitext(file_name)[0]  # Extract date part
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            print(f"Skipping invalid file name: {file_name}")
            continue

    # Identify missing dates
    missing_indices = []
    data = []
    sorted_dates = sorted(dates)

    for i in range(len(sorted_dates) - 1):
        expected_date = sorted_dates[i] + timedelta(days=1)
        if expected_date != sorted_dates[i + 1]:  # Check if next date exists
            missing_indices.append(i + 1)  # Add index of the missing day

    # Load images into the data array
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        try:
            # Load image and resize to a consistent shape (256x256)
            img = Image.open(file_path).convert('RGB')
            img = img.resize((256, 256))  # Resize to 256x256
            data.append(np.array(img))
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    # Normalize pixel values to [0, 1]
    data = np.array(data).astype('float32') / 255.0

    return data, missing_indices, file_names



# Build the Variational Autoencoder (VAE)
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_shape, latent_dim):
    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(16 * 16 * 256, activation='relu')(latent_inputs)
    x = Reshape((16, 16, 256))(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # Loss
    reconstruction_loss = MeanSquaredError()(K.flatten(inputs), K.flatten(outputs))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder, decoder



# Train the VAE
def train_vae(vae, data, epochs=50, batch_size=32):
    history = vae.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Predict Missing Data
def predict_missing_images(encoder, decoder, data, missing_indices):
    filled_data = data.copy()

    for idx in missing_indices:
        # Predict missing image by using the previous available frame
        if idx > 0:
            latent_representation = encoder.predict(filled_data[idx - 1][np.newaxis, :])[2]
            predicted_image = decoder.predict(latent_representation)
            filled_data = np.insert(filled_data, idx, predicted_image, axis=0)

    return filled_data

# Save Missing Images
def save_images(data, missing_indices, file_names, data_dir):
    for idx in missing_indices:
        img_array = (data[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        output_path = os.path.join(data_dir, f"predicted_{idx}.jpeg")
        img.save(output_path)
        print(f"Saved predicted image at {output_path}")


# Save the Model
def save_model(vae, encoder, decoder, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    vae.save(os.path.join(model_path, "vae.h5"))
    encoder.save(os.path.join(model_path, "encoder.h5"))
    decoder.save(os.path.join(model_path, "decoder.h5"))



if __name__ == "__main__":
    data_dir = ".\\sentinel_satellite_images"  
    model_path = ".\\vae_model" 
    latent_dim = 16  # Dimensionality of the latent space

    
    data, missing_indices, file_names = load_image_data(data_dir) # Load and preprocess data    
    vae, encoder, decoder = build_vae(input_shape=data.shape[1:], latent_dim=latent_dim) # Build the VAE
    history = train_vae(vae, data, epochs=50, batch_size=32)     # Train the VAE
    filled_data = predict_missing_images(encoder, decoder, data, missing_indices)  # Predict missing images
    save_images(filled_data, missing_indices, file_names, data_dir)  # Save the predicted images
    save_model(vae, encoder, decoder, model_path) # Save the models
