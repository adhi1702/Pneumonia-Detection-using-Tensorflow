import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Load image with the same target size as the training
    img_array = img_to_array(img) / 255.0  # Convert image to array and scale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the trained model
model = load_model('best_model.keras')  # Change to 'final_model.keras' if you saved the final model

# Predict function
def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    if prediction[0][0] >= 0.5:
        return "PNEUMONIA"
    else:
        return "NORMAL"

# Example usage
image_path = "chest_xray/test/PNEUMONIA/person1682_virus_2899.jpeg"  # Replace with the path to the test image
result = predict_image(image_path)
print(f'The predicted class for the image is: {result}')
