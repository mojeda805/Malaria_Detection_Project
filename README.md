# 🩸 Malaria Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat&logo=opencv)

## 📌 Project Overview
Malaria is a life-threatening disease caused by parasites transmitted to people through the bites of infected mosquitoes. Early and accurate diagnosis is crucial for effective treatment.

This project utilizes **Convolutional Neural Networks (CNN)** to automate the detection of malaria from microscopic blood cell images. By classifying cells as either **Parasitized** or **Uninfected**, this model aims to reduce manual diagnosis time and human error.

## 🎯 Problem Definition
* **Goal:** Build a binary classification model to detect the presence of the *Plasmodium* parasite in blood smear images.
* **Input:** Microscopic RGB images (resized to 64x64).
* **Output:** Class Label (Parasitized vs. Uninfected).

## 📊 Data Description
The dataset contains 27,558 cell images, split into training and testing sets.

* **Total Images:** 27,558
* **Training Set:** 24,958 images
* **Test Set:** 2,600 images
* **Class Balance:** The dataset is well-balanced (~12.5k images per class), mitigating bias issues.



### Preprocessing & EDA
* **Resizing:** All images standardized to `64x64x3` pixels.
* **Normalization:** Pixel values scaled from `[0, 255]` to `[0, 1]`.
* **Encoding:** Labels one-hot encoded for training.
* **Visual Analysis:** Mean image visualization revealed that parasitized cells contain distinct darker regions (the parasite) compared to the clear structure of uninfected cells.
* **Experiments:** Gaussian blurring was tested but discarded as it obscured the sharp edges of the parasites, which are critical features for detection.

## 🧠 Model Architecture & Experiments
Five distinct modeling approaches were evaluated to determine the optimal architecture.



[Image of CNN architecture diagram]


| Model Name | Architecture Summary | Key Techniques | Accuracy (Test) |
| :--- | :--- | :--- | :--- |
| **Base Model** | **3 Conv2D Blocks** | ReLU, MaxPooling, Dropout | **98.35%** |
| **Model 1** | Deep CNN (4 Blocks) | Added Extra Dense Layer (256) | 98.23% |
| **Model 2** | CNN + BatchNorm | BatchNormalization, LeakyReLU | 95.65% |
| **Model 3** | Augmentation CNN | Data Augmentation (Flip, Zoom) | 95.85% (Val) |
| **Model 4** | **VGG16 (Transfer)** | Pre-trained VGG16 + Custom Head | 95.00% |

### Why did the Base Model win?
The **Base Model** (Simple Custom CNN) outperformed complex architectures like VGG16 and models with heavy batch normalization.
* **Hypothesis:** The features required to detect malaria (small dark spots) are relatively simple. Deep, complex networks like VGG16 may have over-abstracted these features or required more data to fine-tune effectively.
* **Efficiency:** The Base Model is computationally cheaper and faster for inference while maintaining the highest accuracy.

## 📉 Key Findings
1.  **Simplicity is Key:** The simpler 3-layer CNN achieved the best results (~98.35%), suggesting that for low-resolution medical imaging (64x64), massive depth is not always necessary.
2.  **Blurring Hurts:** Applying Gaussian Blur degraded performance, confirming that high-frequency details (sharp edges of the parasite) are vital for classification.
3.  **Transfer Learning Limitations:** VGG16 provided decent results (95%) but did not beat the custom model, likely due to the domain gap between ImageNet (objects) and microscopic cells.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Malaria-Detection.git](https://github.com/YOUR_USERNAME/Malaria-Detection.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib opencv-python
    ```
3.  **Run the notebook:**
    Open `Malaria_Detection_Analysis.ipynb` in Jupyter or Google Colab.

## 🔮 Future Work
* **Hyperparameter Tuning:** Automate learning rate searching using Keras Tuner.
* **Advanced Architectures:** Experiment with ResNet50 or EfficientNet for potentially better feature extraction.
* **Explainability:** Implement **Grad-CAM** to visualize exactly *where* the model is looking inside the cell to make a decision.

---
*Created by [Your Name]*
