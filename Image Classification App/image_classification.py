import sys
import io

# Force the standard output to use 'utf-8' encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Base class for handling images (Encapsulation and Inheritance)
class ImageHandler:
    def __init__(self):
        self._image_path = None  # Encapsulation: Private attribute

    def set_image_path(self, path):
        self._image_path = path

    def get_image_path(self):
        return self._image_path

    def load_image(self, path, target_size=(224, 224)):
        img = Image.open(path)
        img = img.resize(target_size)
        return img

# Class for AI-based image classification (Polymorphism and Inheritance)
class ImageClassifier(ImageHandler):
    def __init__(self, model):
        super().__init__()
        self.model = model  # AI model

    # Method to preprocess and classify the image
    def classify_image(self, image_path):
        img = self.load_image(image_path)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = self.model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=3)[0]
        return decoded_preds

# Decorator for logging actions
def action_logger(func):
    def wrapper(*args, **kwargs):
        print(f"Action logged: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Main Tkinter application with image classification functionality
class ImageClassificationApp(tk.Tk, ImageClassifier):
    def __init__(self, model):
        tk.Tk.__init__(self)
        ImageClassifier.__init__(self, model)
        
        self.title("AI Image Classification App")
        self.geometry("500x500")

        # Button to upload image
        self.upload_button = Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        # Label to display the result
        self.result_label = Label(self, text="No Image Uploaded", wraplength=400)
        self.result_label.pack(pady=20)

        # Placeholder for image
        self.image_label = Label(self)
        self.image_label.pack()

    @action_logger  # Decorator for logging action
    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename()
        if file_path:
            self.set_image_path(file_path)
            img = self.load_image(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # Classify the uploaded image
            self.classify_uploaded_image()

    @action_logger
    def classify_uploaded_image(self):
        image_path = self.get_image_path()
        if image_path:
            predictions = self.classify_image(image_path)
            result_text = "Top Predictions:\n"
            for pred in predictions:
                result_text += f"{pred[1]}: {round(pred[2] * 100, 2)}%\n"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="No Image Uploaded")

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Run the Tkinter application
if __name__ == "__main__":
    app = ImageClassificationApp(model)
    app.mainloop()
