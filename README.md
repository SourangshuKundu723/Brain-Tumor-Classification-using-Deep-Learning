# 🧠 Medical Image Analysis using Deep Learning (Brain Tumor Detection)

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Unbounded&weight=800&size=28&duration=1500&pause=600&color=FF6F00&center=true&vCenter=true&width=900&lines=🚧+Work+In+Progress;🧠+Brain+Tumor+Classification+Model+Training..." />
<br>
<br>

![Repo Views](https://komarev.com/ghpvc/?username=SourangshuKundu723&repo=Brain-Tumor-Classification-using-Deep-Learning&color=blue&style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/SourangshuKundu723/Brain-Tumor-Classification-using-Deep-Learning)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aJplzvtxr-RN08TLUsebfw7l6qNLTZhN?usp=sharing)

</div>

This project focuses on **Brain Tumor Detection** from MRI images using **Deep Learning** techniques built with **TensorFlow** and **Keras**.  
It leverages **MobileNet** for efficient image classification, making it suitable for both research and deployment.


## 🚀 Project Overview

The goal is to build a **Convolutional Neural Network (CNN)** that can classify brain MRI images into **four categories**:
- **Glioma tumor**
- **Meningioma tumor**
- **Pituitary tumor**
- **Normal**.  

The model is trained, validated, and tested on MRI image datasets and can also be used to predict the class of new uploaded images.

## 🧩 Features

- Google Colab–ready code for easy execution  
- Organized modular structure (`src/` folder for maintainability)  
- Uses **MobileNet** as base CNN model  
- Includes:
  - Data preprocessing and augmentation  
  - Model training with checkpoints and early stopping  
  - Visualization of training progress  
  - Model evaluation (accuracy, confusion matrix, classification report)  
  - Single image prediction interface  

## 🧑‍💻 Technologies Used

- **Deep Learning Framework:** TensorFlow, Keras  
- **Image Processing:** OpenCV, PIL (Python Imaging Library)  
- **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn  
- **Model Evaluation:** Scikit-learn (classification report, confusion matrix)  
- **Development Environment:** Google Colab, VS Code  
- **Others:** Logging, Exception Handling, Model Checkpointing, Early Stopping


## 🧠 Model Architecture

- **Base Model:** MobileNet (pretrained on ImageNet)
- **Input Size:** 128 × 128 × 3
- **Loss:** Categorical Crossentropy    
- **Metrics:** Accuracy  

## 🧬 Dataset

The dataset used is the Brain Tumor MRI Dataset available on **Kaggle**. It contains MRI images for training and testing the model.

🔗 [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/babaraliuser/brain-mri-images-dataset)

## ⚙️ Installation & Setup

You can run this project directly in **Google Colab**. Click below to open the notebook:


<div align = "center">
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aJplzvtxr-RN08TLUsebfw7l6qNLTZhN?usp=sharing)
<br>
</div>

If you prefer to run it locally (e.g., in VS Code), install the dependencies using:

```bash
pip install tensorflow keras opencv-python Pillow numpy pandas matplotlib seaborn scikit-learn tqdm
```
## 🧪 Evaluation

The model was evaluated on the test dataset containing four classes — **Glioma tumor**, **Meningioma tumor**, **Pituitary tumor**, and **Normal**.  

A **confusion matrix** and **classification report** were used to analyze class-wise accuracy, precision, and recall.

## 🙏 Acknowledgements

- The dataset is provided by [BABAR ALI](https://www.kaggle.com/datasets/babaraliuser/brain-mri-images-dataset) on Kaggle.

- The project uses **TensorFlow**, **Keras**, and **Google Colab** for training and deployment

💡 **"Deep learning models may not replace doctors, but they can greatly assist in faster and more accurate medical diagnosis."**