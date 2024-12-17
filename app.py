from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the Keras model
model = tf.keras.models.load_model('model1.keras')
label_map = pd.read_csv("input/emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None,
                        )
label_m = label_map.squeeze()
 # Map prediction to character (assuming you have a label mapping)
label_dict={}
for i,labels in enumerate(label_m):
    label_dict[i] = chr(labels)
label_dict

def preprocess_image(image_path):
    # Image preprocessing similar to training
    image = Image.open(image_path)
    image = image.resize((28, 28))  # Match training dimensions
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image = image / 255.0  # Normalize pixel values
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['imagefile']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Save the uploaded image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Preprocess image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)

            
            result = label_dict[predicted_class]
            
            return render_template('index.html', prediction=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)