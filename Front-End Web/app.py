from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
dic = {0:'Acne and Rosacea',1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 2:'Atopic Dermatitis',3: 'Bullous Disease',4: 'Cellulitis Impetigo and other Bacterial Infections',5: 'Eczema',6: 'Exanthems and Drug Eruptions',7: 'Hair Loss Photos Alopecia and other Hair Diseases',8: 'Herpes HPV and other STDs',9: 'Light Diseases and Disorders of Pigmentation',10: 'Lupus and other Connective Tissue diseases',11: 'Melanoma Skin Cancer Nevi and Moles',12: 'Nail Fungus and other Nail Disease',13: 'Poison Ivy Photos and other Contact Dermatitis',14: 'Psoriasis pictures Lichen Planus and related diseases',15: 'Scabies Lyme Disease and other Infestations and Bites',16: 'Seborrheic Keratoses and other Benign Tumors',17: 'Systemic Disease',18: 'Tinea Ringworm Candidiasis and other Fungal Infections',19: 'Urticaria Hives',20: 'Vascular Tumors',21: 'Vasculitis Photos',22: 'Warts Molluscum and other Viral Infections'}
model = tf.keras.models.load_model('model.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect():
    return render_template('detect.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['image']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    result = model.predict(image)
    index=np.argmax(result[0])
    classification = '%s' % (dic[index])

    return render_template('detect.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

