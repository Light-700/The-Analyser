# from flask import Flask, request, render_template, jsonify
# import os
# import pickle
# from PIL import Image
# import numpy as np
# # from flask_cors import CORS
# # import os

# app = Flask(__name__)
# # CORS(app)

# # UPLOAD_FOLDER = 'uploads'
# # if not os.path.exists(UPLOAD_FOLDER):
#     # os.makedirs(UPLOAD_FOLDER)

# @app.route('/upload', methods=['GET'])
# def hello_world():
#     # return "Hello, World!"
#     return render_template('index_.html')
# # def upload_file():
#     # if 'file' not in request.files:
#         # return jsonify({'error': 'No file part'}), 400
#     # file = request.files['file']
#     # if file:
#         # filename = file.filename
#         # file.save(os.path.join(UPLOAD_FOLDER, filename))
#         # return jsonify({'message': 'File uploaded successfully'})

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_path = "./uploads/" + imagefile.filename
#     imagefile.save(image_path)

#     # image = load_img(image_path, target_size=(224, 224))
#     # image = img_to_array(image)
#     # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # image = preprocess_input(image)
#     # yhat = model.predict(image)
#     # label = decode_predictions(yhat)
#     # label = label[0][0]

#     # classification = '%s (%.2f%%)' % (label[1], label[2]*100)

#     return render_template('index_.html', prediction="Prediction: " + imagefile.filename)

# if __name__ == '__main__':
#     # app.run(port=os.getenv('PORT', 5000))
#     app.run(port=5000, debug=True)


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
label_map = pd.read_csv("/The-Analyser/input/emnist-balanced-mapping.txt", 
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
            
            return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)