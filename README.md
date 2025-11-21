# Face Mask Detection App 😷

A complete Face Mask Detection application using Python, TensorFlow, MobileNetV2, Mediapipe, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Live Webcam Detection**: Real-time face mask detection using your webcam
- **Image Upload**: Detect masks in uploaded images (JPG/PNG)
- **Video Upload**: Process video files for mask detection (MP4/AVI)
- **Training Pipeline**: Complete scripts to train a custom model
- **MobileNetV2 Architecture**: Fast and accurate transfer learning model
- **Mediapipe Face Detection**: Robust face detection in various conditions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Mayank-iitj/mask.git
    cd mask
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Dataset**
    
    **Option A: Download Real Dataset (Recommended)**
    ```bash
    # Install kaggle API
    pip install kaggle
    
    # Configure kaggle credentials (get from https://www.kaggle.com/account)
    # Place kaggle.json in ~/.kaggle/ or C:\Users\<username>\.kaggle\
    
    # Run download script
    python utils/download_dataset.py
    ```
    
    **Option B: Use Dummy Data (for testing)**
    ```bash
    python utils/create_dummy_data.py
    ```
    
    **Option C: Manual Dataset**
    - Download from: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
    - Extract and organize in `data/train`, `data/val`, `data/test`
    - Each folder should have `mask` and `no_mask` subfolders

## 📖 Usage

### Run the App
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Train the Model
```bash
python training/train_model.py
```

### Evaluate the Model
```bash
python training/evaluate_model.py
```

## 🏗️ Project Structure

```
mask/
├── app.py                      # Main Streamlit application
├── mask_config.py              # Configuration settings
├── requirements.txt            # Python dependencies
├── models/
│   ├── model_builder.py        # MobileNetV2 model architecture
│   └── mask_detector.h5        # Trained model (after training)
├── utils/
│   ├── data_loader.py          # Data loading and generators
│   ├── preprocessing.py        # Image preprocessing
│   ├── inference.py            # Face detection and classification
│   ├── visualization.py        # Drawing utilities
│   ├── create_dummy_data.py    # Dummy dataset generator
│   ├── download_dataset.py     # Dataset download script
│   └── organize_dataset.py     # Dataset organization script
├── training/
│   ├── train_model.py          # Training script
│   └── evaluate_model.py       # Evaluation script
└── data/                       # Dataset directory
    ├── train/
    ├── val/
    └── test/
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.mask.yml up -d

# Access the app at http://localhost:8501
```

## ☁️ Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions for:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Run

## 🧠 Model Details

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224x224 RGB
- **Classes**: 2 (Mask, No Mask)
- **Face Detection**: Mediapipe Face Detection
- **Framework**: TensorFlow/Keras

## 📊 Dataset

The app supports any face mask dataset organized as:
```
data/
├── train/
│   ├── mask/
│   └── no_mask/
├── val/
│   ├── mask/
│   └── no_mask/
└── test/
    ├── mask/
    └── no_mask/
```

Recommended dataset: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mayank**
- GitHub: [@Mayank-iitj](https://github.com/Mayank-iitj)

## 🙏 Acknowledgments

- MobileNetV2 architecture from TensorFlow
- Mediapipe for face detection
- Streamlit for the web interface
- Face Mask Detection Dataset from Kaggle
