from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model/lung_multiclass_model.keras")

class_labels = {
    0: "Benign cases",
    1: "Malignant cases",
    2: "Adenocarcinoma",
    3: "Large Cell",
    4: "Normal",
    5: "Squamous",
}

def predict_lung(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)
    idx = np.argmax(pred)

    return class_labels[idx], float(np.max(pred)*100)