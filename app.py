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
import pickle
from PIL import Image
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
with open('model1.keras', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['imagefile']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Save and process image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Preprocess image for keras model
            image = Image.open(filepath)
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            # Make prediction
            preds = model.predict(image)
            results = decode_predictions(preds, top=1)[0]
            prediction = results[0][1]
            
            return render_template('index_.html', prediction=prediction)
    
    return render_template('index_.html')

if __name__ == '__main__':
    app.run(debug=True)