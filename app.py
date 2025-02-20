import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, redirect

# Load pre-trained model
load_model = tf.keras.models.load_model
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "static/images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load your trained model
model = load_model("googlenet_keras.h5")  # Ensure this model exists

# Preprocessing function (resize image to 224x224)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/blog")
def blog():
    return redirect("https://pmc.ncbi.nlm.nih.gov/articles/PMC9891061/")

@app.route("/news")
def news():
    return redirect("https://www.medicalnewstoday.com/")


# Flask route for file upload and prediction
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!"
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file!"

        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Process and predict
        img = preprocess_image(file_path)
        prediction = model.predict(img)

        # Corrected classification logic
        predicted_class_index = np.argmax(prediction)  # Get highest probability class
        class_labels = ["Normal", "Monkeypox"]  # Adjust based on training labels
        predicted_class = class_labels[predicted_class_index]
        confidence = round(float(np.max(prediction)) * 100, 2)  # Get confidence score

        return render_template("result.html", image=file_path, label=predicted_class, confidence=confidence)

    return render_template("upload.html")  # Create an HTML upload form

if __name__ == "__main__":
    app.run(debug=True)
