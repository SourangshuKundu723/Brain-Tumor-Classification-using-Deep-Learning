from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model/brain_mri_finetuned_persistent.keras")

class_labels = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

def predict_brain(img_path):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)

    return class_labels[idx], float(np.max(pred)*100)