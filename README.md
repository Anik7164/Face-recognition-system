# Facial Expression Recognition System

A real-time facial expression recognition system using Convolutional Neural Networks (CNN) with TensorFlow and Keras. The system can detect and classify 7 different emotions from live webcam feed.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Features](#features)
- [Emotion Classes](#emotion-classes)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Real-Time Detection](#real-time-detection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Real-time emotion detection** from webcam feed
- **7 emotion classifications**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Face detection** using Haar Cascade Classifier
- **Prediction smoothing** for stable results
- **Visual confidence indicators** with color-coded emotions
- **Confidence bars** and probability distributions
- **Screenshot capture** functionality
- **Grayscale image processing** for better performance

## Emotion Classes

The model recognizes the following 7 emotions:

1. **Angry** (Red)
2. **Disgust** (Green)
3. **Fear** (Purple)
4. **Happy** (Yellow)
5. **Neutral** (White)
6. **Sad** (Blue)
7. **Surprise** (Orange)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- GPU (optional, for faster training)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Anik7164/Face-recognition-system.git
cd Face-recognition-system
```

2. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python --version
pip list
```

## Project Structure

```
Face-recognition-system/
│
├── main.py                          # Main application file (real-time detection)
├── facial_expression_model.h5       # Pre-trained model
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── LICENSE                          # License file
│
├── data/                            # Dataset directory
│   ├── Train/                       # Training data
│   │   ├── Anger/
│   │   ├── Contempt/
│   │   ├── Disgust/
│   │   ├── Fear/
│   │   ├── Happy/
│   │   ├── Neutral/
│   │   ├── Sad/
│   │   └── Surprised/
│   │
│   └── Validation/                  # Validation data
│       ├── Anger/
│       ├── Contempt/
│       ├── Disgust/
│       ├── Fear/
│       ├── Happy/
│       ├── Neutral/
│       ├── Sad/
│       └── Surprised/
│
└── support/                         # Supporting code files
    ├── train.py                     # Model training script
    ├── a.py                         # Dataset verification script
    └── *.ipynb                      # Jupyter notebooks for experiments
```

## Usage

### Real-Time Emotion Detection

Run the main application to start real-time emotion detection:

```bash
python main.py
```

**Keyboard Controls:**

- Press `q` to quit the application
- Press `r` to reset emotion history
- Press `s` to save current frame as screenshot

### Training the Model

If you want to train the model from scratch or retrain with new data:

```bash
python support/train.py
```

Make sure to update the dataset paths in the training script before running.

## Dataset

The model is trained on a facial expression dataset with 8 emotion categories:

- Anger
- Contempt
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

### Dataset Structure

- Images are **48x48 pixels** in **grayscale**
- Organized in folders by emotion class
- Split into **Training** and **Validation** sets

### Data Augmentation

The training process uses:

- Pixel value normalization (0-1 range)
- Automatic resizing to 48x48
- Grayscale conversion

## Model Architecture

The CNN model consists of:

```
Input Layer (48x48x1)
    ↓
Conv2D (32 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Dropout (0.25)
    ↓
Flatten
    ↓
Dense (1024 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (8 units, Softmax)
```

**Key Features:**

- **Input**: 48x48 grayscale images
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Total Parameters**: ~8M trainable parameters

## Training

### Training Configuration

- **Batch Size**: 64
- **Epochs**: 100
- **Image Dimensions**: 48x48 pixels
- **Color Mode**: Grayscale
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy

### Training Process

1. Data is loaded from the `data/` directory
2. Images are automatically resized and normalized
3. Model trains using categorical crossentropy loss
4. Best model is saved as `facial_expression_model.h5`

## Real-Time Detection

The real-time detection system:

1. Captures video from webcam
2. Detects faces using Haar Cascade
3. Extracts face ROI and preprocesses it
4. Predicts emotion using the trained model
5. Applies temporal smoothing (15-frame history)
6. Displays results with color-coded bounding boxes
7. Shows confidence bars and probability distributions

### Performance Features

- **Prediction Smoothing**: Uses 15-frame history for stable predictions
- **Multi-face Detection**: Can detect multiple faces simultaneously
- **Visual Feedback**: Color-coded emotions with confidence scores
- **Real-time Processing**: Optimized for smooth video processing

## Results

The model achieves competitive accuracy on facial expression recognition tasks:

- Real-time processing at 30+ FPS on modern hardware
- Accurate emotion detection in various lighting conditions
- Robust face detection with Haar Cascade classifier
- Smooth predictions with temporal averaging

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- Add more emotion classes
- Improve model accuracy
- Add support for batch image processing
- Implement emotion tracking over time
- Add support for video file input
- Optimize for mobile deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Anik7164**

- GitHub: [@Anik7164](https://github.com/Anik7164)
- Repository: [Face-recognition-system](https://github.com/Anik7164/Face-recognition-system)

## Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- OpenCV community for computer vision tools
- Dataset contributors for facial expression data
- Research community for CNN architectures and techniques

## Support

If you have any questions or issues:

- Open an issue on GitHub
- Check existing issues for solutions
- Review the documentation

## Future Enhancements

- [ ] Add support for multiple face tracking
- [ ] Implement emotion history timeline
- [ ] Add data augmentation for training
- [ ] Create web interface with Flask/Django
- [ ] Add emotion-based music recommendation
- [ ] Implement transfer learning with pre-trained models
- [ ] Add mobile app support (Android/iOS)
- [ ] Create REST API for emotion detection

---

If you find this project useful, please consider giving it a star!
