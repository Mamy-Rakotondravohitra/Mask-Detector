
from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy
import os
import cv2

app = Flask(__name__)

#Chargement des modèles
@app.before_first_request
def load__model():
 '''
 Load model
 :return: model (global variable)
 '''
 print('[INFO] Model Loading ……..')
 global model, face_cascade
 #Modèle de classification du masque
 model = load_model('static/models/InceptionV3_75_mask_30.h5')
 #Modèle de détection des visages
 face_cascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')
 print('[INFO] : Models loaded')

@app.route('/predict', methods=['POST'])
def predict():
    # Import de l'image
    file = request.files['ImgUpload']
    fullname = os.path.join('static/upload/', file.filename)
    file.save(fullname)
    img =cv2.imread(fullname)

    # Conversion en nuances de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détection du ou des visages de l'image chargée
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Traçage des lines autour de chaque visage selon la prédiction du modèle
    i = 0
    j = 0
    for (x, y, w, h) in faces:
        crop_img = img[y:int(round(y+h*1.2)), x:int(round(x+w*1.2))]
        face = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (192, 192))
        img_tf = tf.constant(face)
        img_rezize = img_tf/255
        img_rezize = tf.expand_dims(img_rezize, axis=0)
        prediction = model.predict_classes(img_rezize.numpy())
        pred_number = model.predict(img_rezize.numpy())

        if prediction[0]==0:
            j = j+1
            pred_mask = "No mask"
            cv2.rectangle(img, (x, y), (int(round(x+w)), int(round(y+h))), (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (int(round(x+w)), int(round(y-20))), (0, 0, 255), -1)
            cv2.putText(img, pred_mask,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
        else :
            i = i+1
            pred_mask = "Mask"
            cv2.rectangle(img, (x, y), (int(round(x+w)), int(round(y+h))), (84, 222, 114), 2)
            cv2.rectangle(img, (x, y), (int(round(x+w)), int(round(y-20))), (84, 222, 114), -1)
            cv2.putText(img,pred_mask, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
    # Prediction
    #fullname2 = os.path.join('static/upload/pred_test.jpg', file.filename)
    fullname2 = os.path.join('static/upload/', 'result_'+file.filename)
    cv2.imwrite(fullname2,img)
    #fullname2 = os.path.join('static/upload/', "pred_test.jpg")
    return  render_template("predicted.html",nb_mask = i, nb_no_mask = j, img_path = fullname2, result = fullname2)

@app.route('/simulation', methods=['GET'])
def simulation():
    return render_template('simulation.html')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
