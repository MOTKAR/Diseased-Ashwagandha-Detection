# Ashwagandha Disease Detection Model

A deep learning-based ensemble model for automated classification and detection of diseases in Ashwagandha plant leaves.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results & Evaluation](#results--evaluation)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## 🎯 Problem Statement

Ashwagandha (Withania somnifera) is a highly valued medicinal plant in traditional medicine systems. Early detection of diseases in Ashwagandha plants is crucial for maintaining crop health and ensuring quality yield. Manual inspection is time-consuming, labor-intensive, and prone to human error.

This project addresses this challenge by developing an automated deep learning model that can accurately classify Ashwagandha leaves as either **Healthy** or **Diseased** using image analysis and transfer learning techniques.

---

## ✨ Features

- **Ensemble Architecture**: Combines three pre-trained convolutional neural networks (VGG16, ResNet50, InceptionV3)
- **Transfer Learning**: Leverages ImageNet pre-trained weights for improved accuracy
- **Automated Classification**: Binary classification of Ashwagandha leaf images (Healthy/Diseased)
- **Comprehensive Evaluation**: Confusion matrix, precision, recall, F1-score, and ROC-AUC metrics
- **High Performance**: Macro-averaged evaluation metrics for balanced multi-class assessment
- **Model Persistence**: Trained model saved in HDF5 format for deployment
- **Data Augmentation**: Built-in data augmentation with validation split for robust training

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: TensorFlow/Keras
- **Pre-trained Models**: VGG16, ResNet50, InceptionV3 (from Keras Applications)
- **Data Processing**: NumPy
- **Image Processing**: Keras ImageDataGenerator
- **Evaluation Metrics**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Language**: Python 3.8+

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 4GB+ RAM recommended
- GPU support recommended (NVIDIA CUDA)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Diseased-Ashwagandha-Detection-Model.git
cd Diseased-Ashwagandha-Detection-Model
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Your Dataset
Create a folder structure as follows:
```
Dataset/
├── Healthy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Diseased/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Ensure all images are in a supported format (JPG, PNG, etc.) with appropriate resolution.

---

## 🚀 Usage

### Training the Model

1. Open the Jupyter Notebook:
```bash
jupyter notebook final_code.ipynb
```

2. Execute the cells sequentially:
   - **Cell 1**: Data preparation and model definition
   - **Cell 2**: Model training (8 epochs)
   - The trained model will be saved as `ashwagandha_stacked_model.h5`

### Making Predictions

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('ashwagandha_stacked_model.h5')

# Load and preprocess an image
img = load_img('path/to/leaf/image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
class_label = 'Healthy' if prediction[0][0] > prediction[0][1] else 'Diseased'
confidence = max(prediction[0]) * 100

print(f"Classification: {class_label}")
print(f"Confidence: {confidence:.2f}%")
```

### Evaluating the Model

Run the evaluation cells to get:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Accuracy Score
- ROC-AUC curves
- Performance Metrics Visualization

---

## 📂 Project Structure

```
Diseased-Ashwagandha-Detection-Model/
│
├── final_code.ipynb              # Main Jupyter notebook with complete pipeline
│   ├── Cell 1: Data Preparation & Model Training
│   ├── Cell 2: Confusion Matrix & Classification Metrics
│   ├── Cell 3: Detailed Performance Evaluation
│   └── Cell 4: Metric Visualization
│
├── requirements.txt              # Project dependencies
├── README.md                      # Project documentation
├── LICENSE                        # GNU General Public License v3
│
└── Dataset/                       # Image dataset (not included)
    ├── Healthy/
    └── Diseased/
```

---

## 🧠 Model Architecture

### Ensemble Strategy
The model combines three powerful pre-trained CNN architectures:

1. **VGG16**: Deep convolutional network with 16 layers
   - Feature extraction: Global Average Pooling → Flattening
   
2. **ResNet50**: Residual network with 50 layers
   - Feature extraction: Global Average Pooling → Flattening
   
3. **InceptionV3**: Multi-scale convolutional network
   - Feature extraction: Global Average Pooling → Flattening

### Feature Fusion & Classification
```
Input Image (224×224×3)
        ↓
[VGG16] [ResNet50] [InceptionV3]  ← Parallel Feature Extraction
        ↓           ↓           ↓
    Flatten     Flatten     Flatten
        ↓___________↓___________|
            Concatenate
                ↓
        Dense(512, relu) → Dropout(0.5)
                ↓
        Dense(256, relu) → Dropout(0.5)
                ↓
        Dense(num_classes, softmax)
                ↓
        Output: Healthy/Diseased
```

### Training Configuration
- **Input Size**: 224×224×3 (RGB images)
- **Batch Size**: 32
- **Epochs**: 8
- **Validation Split**: 20%
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## 📊 Results & Evaluation

### Performance Metrics
The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: True positive rate among positive predictions
- **Recall/Sensitivity**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity (TNR)**: True negative rate among actual negatives
- **False Positive Rate (FPR)**: False alarm rate
- **Confusion Matrix**: Detailed classification breakdown

### Example Output
```
=========== MODEL PERFORMANCE METRICS ===========
Accuracy:               92.50 %
Precision (Macro):      91.23 %
Recall / Sensitivity:   90.85 %
F1 Score (Macro):       91.04 %
Specificity (TNR):      93.12 %
True Positive Rate:     90.85 %
False Positive Rate:     6.88 %
Misclassification:       7.50 %
```

### Visualizations
- Confusion Matrix Heatmap
- Classification Report Table
- Accuracy/Precision/F1-Score Bar Chart
- Performance Metrics Line Plot

---

## 🔮 Future Improvements

1. **Real-time Inference**
   - Develop a web-based or mobile application for real-time leaf classification
   - Add camera integration for instant disease detection

2. **Multi-class Disease Classification**
   - Extend to detect specific types of diseases (powdery mildew, leaf spot, etc.)
   - Include multiple severity levels

3. **Model Optimization**
   - Implement quantization for edge deployment
   - Convert to TensorFlow Lite for mobile applications
   - Explore knowledge distillation for lightweight models

4. **Data Collection & Augmentation**
   - Expand dataset with more diverse Ashwagandha varieties
   - Include images from different environmental conditions and growth stages
   - Implement advanced augmentation techniques (Mixup, CutOut, etc.)

5. **Explainability & Interpretability**
   - Integrate Grad-CAM for visualization of important regions in the leaf
   - Provide explainable AI insights for farmer understanding

6. **Deployment Solutions**
   - REST API with Flask/FastAPI for cloud integration
   - Docker containerization for easy deployment
   - Integration with IoT sensors for automated monitoring

7. **Enhanced Preprocessing**
   - Implement leaf segmentation for better feature extraction
   - Add color correction for varying lighting conditions
   - Develop preprocessing pipelines for different image qualities

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**Swapnil Motkar**  
MIT WORLD PEACE UNIVERSITY  
📧 Email: motkarswapnil@gmail.com 

**Om Patni**
MIT WORLD PEACE UNIVERSITY
📧 Email: ompatni1908@gmail.com

**Pranav Mode**
MIT WORLD PEACE UNIVERSITY
📧 Email: pranav16mode@gmail.com

**Sakshi Naikwade**
MIT WORLD PEACE UNIVERSITY
📧 Email: sakshinaikwade18@gmail.com

**Diseased Ashwagandha Detection Model**

Created for automated agricultural disease detection and crop health monitoring.

### Contact & Contributions
- For issues and contributions, please refer to the repository's issue tracker
- Suggestions and improvements are welcome!

---

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Keras Applications for pre-trained models (VGG16, ResNet50, InceptionV3)
- ImageNet dataset for transfer learning capabilities
- Scikit-learn for evaluation metrics

---

**Note**: Ensure you have appropriate permissions to use and distribute any datasets used with this model. This model is intended for educational and research purposes.
