from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/button_click', methods = ['POST'])
def run_model():
    data = request.get_json()
    button_id = data['button_id']

    if button_id == 'run_model':
        width = 180
        height = 180
        ds_path = 'Fruit_dataset_v2/train'
        train_img_dataset = tf.keras.preprocessing.image_dataset_from_directory(ds_path, validation_split = 0.2, subset = "training", seed = 123, labels = "inferred", color_mode = 'rgb', image_size = (width, height),  batch_size = 32)
        class_names = train_img_dataset.class_names

        model = tf.keras.models.load_model("Fruit_Quality_Classification_Model.keras")
        
        img = Image.open("image1.png")
        new_image = img.resize((180, 180))
        new_img_rgb = new_image.convert('RGB')
        img_array = np.array(new_img_rgb)
        predictions = model.predict(img_array[None,:,:,:])
        predicted_index = np.argmax(predictions, axis = 1)

        response = f"A {class_names[predicted_index[0]]} was detected"

    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=False)

import tensorflow as tf
rec_model = tf.keras.models.load_model("Fruit_Quality_Classification_Model.keras")
