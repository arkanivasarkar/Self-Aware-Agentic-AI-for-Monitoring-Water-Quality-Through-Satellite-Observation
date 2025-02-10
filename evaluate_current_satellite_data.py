import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import pipeline

# Function to preprocess image and extract features
def extract_image_features(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array


# Function to predict using the saved models for today's date
def predict_today(image_folder, target):
    model = load_model(f"{target}_lstm_model.h5")

    # Get today's date
    today = dt.datetime.today().strftime("%Y-%m-%d")
    image_path = os.path.join(image_folder, f"{today}.png")
    
    # Extract features for the image corresponding to today
    img_array = extract_image_features(image_path)
    img_data = np.expand_dims(img_array, axis=0)  # Reshape to match LSTM input

    # Make prediction
    img_data = img_data.reshape((1, 30, -1)) 
    prediction = model.predict(img_data)

    return prediction


# Function to describe the regression plot
def describe_plot(prediction, target):
    mean_value = np.mean(prediction)
    trend = "increasing" if np.mean(np.diff(prediction)) > 0 else "decreasing"
    peaks = np.max(prediction)
    
    description = f"The predicted {target} levels over time show a {trend} trend. " \
                  f"The mean value of the predicted levels is {mean_value:.2f}. " \
                  f"The highest observed value is {peaks:.2f}. "
    if peaks > 80:
        description += "The peak levels suggest the possibility of algal bloom or other water quality issues."
    elif mean_value > 70:
        description += "The mean levels suggest that the water quality might be affected."
    else:
        description += "The water quality appears to be within acceptable limits."
    return description


# Function to interact with LLaMA to analyze the plot description
def analyze_with_llama(description):
    llama = pipeline("text-generation", model="meta-llama/Llama-2-7b") 
    result = llama(description, max_length=200) 
    response = result[0]['generated_text']
    return response


def monitor_water_quality(location):
    targets = ["chlorophyll", "algal_bloom", "dissolved_oxygen"] 
    description = ''
    for target in targets:
        image_folder = f"data/{location}/{target}"
        
        # Step 1: Generate prediction for today's image
        prediction = predict_today(image_folder, target)
        
        # Step 2: Create a plot description
        description += describe_plot(prediction, target)
        
    # Step 3: Use LLaMA to analyze the description and decide if water quality is bad
    result = analyze_with_llama(description)
    return result
