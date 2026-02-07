# FitnessAI

An AI-powered fitness application that uses computer vision and deep learning to analyze and classify exercise poses in real-time. The system leverages MediaPipe for pose detection and custom-trained neural networks to provide feedback on exercise form.

## 🎯 Project Overview

FitnessAI is designed to help users perform exercises with correct form by analyzing their body positions through image/video processing. The application currently focuses on plank exercises, classifying poses into three categories:
- **Correct**: Proper plank form
- **Incorrect-Low**: Hips too low
- **Incorrect-High**: Hips too high

## ✨ Key Features

- **Real-time Pose Detection**: Uses MediaPipe's pose landmark detection to extract 33 body keypoints
- **Exercise Classification**: Deep learning models trained to classify exercise correctness
- **Multi-Model Approach**: Supports both CNN (image-based) and traditional ML (keypoint-based) models
- **Data Preprocessing Pipeline**: Automated extraction and processing of pose keypoints from images
- **Model Evaluation**: Comprehensive testing and validation framework


## 🔧 Technical Stack

### Core Technologies
- **MediaPipe**: Pose landmark detection and tracking
- **TensorFlow/Keras**: Deep learning model development
- **OpenCV**: Image processing and visualization
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Data preprocessing and evaluation metrics

### Key Dependencies
```python
mediapipe==0.10.11
tensorflow
pandas
numpy
opencv-python
pillow
joblib
```

## 📊 Data Processing Pipeline

### 1. Pose Keypoint Extraction
The `Preprocessing_data.ipynb` notebook handles:
- Loading and processing exercise images
- Detecting pose landmarks using MediaPipe
- Extracting 17 key body points (Nose, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Heels, Foot Indices)
- Storing keypoints with X, Y, Z coordinates in CSV format

### 2. Model Training
The `deep_learning/CNN_plank.ipynb` notebook implements:


- **Keypoint-Based Features**:
  - 17 body landmarks × 3 coordinates (X, Y, Z) = 51 features
  - Normalized coordinates for scale invariance
  - StandardScaler preprocessing

- **CNN Architecture**:
  - Input: 224x224x3 RGB images
  - 2 Convolutional layers (32 and 64 filters)
  - MaxPooling layers
  - Dense layers with dropout (0.3)
  - Softmax output for 3-class classification

- **Training Strategy**:
  - Stratified K-Fold cross-validation (6 folds)
  - 5 folds for training/validation
  - 1 fold reserved for final testing
  - Class weight balancing
  - Early stopping and learning rate reduction

- **Performance Metrics**:
  - Precision, Recall, F1-Score
  - Confusion matrices
  - Per-class performance analysis

### 3. Model Testing
The `ResultTesting.ipynb` notebook provides:
- Loading pre-trained models
- Real-time inference on new images
- Visualization of predictions with confidence scores
- Pose landmark overlay on images

## 🚀 Getting Started

### Prerequisites
```bash
pip install mediapipe==0.10.11 tensorflow pandas numpy opencv-python pillow joblib
```

### Download Required Files
The MediaPipe pose landmark model is automatically downloaded:
```python
!wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

### Usage

#### 1. Preprocess Training Data
```python
# Run Preprocessing_data.ipynb
# - Processes images from your dataset
# - Extracts pose keypoints
# - Saves to CSV for model training
```

#### 2. Train the Model
```python
# Run deep_learning/CNN_plank.ipynb
# - Trains CNN model on processed images
# - Performs cross-validation
# - Saves best model and scaler
```

#### 3. Test on New Images
```python
# Run ResultTesting.ipynb
# - Load trained model
# - Process new exercise images
# - Get classification results with confidence
```

## 📈 Model Performance

The CNN model achieves:
- **High accuracy** on validation sets (typically >95%)
- **Balanced performance** across all three classes
- **Robust predictions** with confidence scores

Example metrics from training:
- Validation Accuracy: ~96%
- F1 Score: ~0.96
- Precision/Recall: Balanced across classes

## 🔮 Future Enhancements

- [ ] Support for additional exercises (squats, push-ups, etc.)
- [ ] Real-time video processing
- [ ] Mobile application deployment
- [ ] Personalized feedback and recommendations
- [ ] Progress tracking and analytics
- [ ] Multi-user support



## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- **MediaPipe**: For the excellent pose detection framework
- **TensorFlow**: For the deep learning infrastructure
- **Google Colab**: For providing computational resources

---

**Note**: This project is designed to run in Google Colab environments. Some paths and configurations may need adjustment for local execution.
