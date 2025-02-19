import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
load_model=tf.keras.models.load_model

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = "static/images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the pre-trained model (GoogLeNet or your model)
model = load_model("googlenet_keras.h5")  # Change this to your model file

# Preprocessing function (resize image to 224x224)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to model's expected input
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Flask route for file upload and prediction
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        # Save the uploaded file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Process and predict
        img = preprocess_image(file_path)
        prediction = model.predict(img)

         # Get predicted class and confidence
        predicted_class = np.argmax(prediction)  # Change based on your model
        confidence = round(np.max(prediction) * 100, 2)

        return render_template("result.html", image=file_path, label=predicted_class, confidence=confidence)


    return render_template("upload.html")  # Create an HTML upload form

if __name__ == "__main__":
    app.run(debug=True)
