# Deployment Guide

## Local Deployment

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/Mayank-iitj/mask.git
   cd mask
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset** (choose one)
   - Use dummy data: `python utils/create_dummy_data.py`
   - Download real dataset: Follow instructions in README.md
   - Use your own dataset in `data/train`, `data/val`, `data/test`

5. **Train the model**
   ```bash
   python training/train_model.py
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Access the app**
   - Open browser: http://localhost:8501

---

## Docker Deployment

### Prerequisites
- Docker
- Docker Compose

### Steps
1. **Build and run with Docker Compose**
   ```bash
   docker-compose -f docker-compose.mask.yml up -d
   ```

2. **Access the app**
   - Open browser: http://localhost:8501

3. **Stop the application**
   ```bash
   docker-compose -f docker-compose.mask.yml down
   ```

---

## Cloud Deployment

### Streamlit Cloud (Free)

1. **Push to GitHub** (already done)

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Mayank-iitj/mask`
   - Main file path: `app.py`
   - Click "Deploy"

**Note**: You'll need to train the model locally and commit the `.h5` file (or use a pre-trained model URL).

### Heroku

1. **Create `Procfile`**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**
   ```bash
   heroku create face-mask-detector
   git push heroku main
   ```

### AWS EC2

1. **Launch EC2 instance** (Ubuntu)
2. **SSH into instance**
3. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   ```
4. **Clone and setup** (follow Local Deployment steps)
5. **Run with nohup**
   ```bash
   nohup streamlit run app.py --server.port=8501 &
   ```

### Google Cloud Run

1. **Build Docker image**
   ```bash
   docker build -f Dockerfile.mask -t face-mask-app .
   ```

2. **Tag and push to GCR**
   ```bash
   docker tag face-mask-app gcr.io/PROJECT_ID/face-mask-app
   docker push gcr.io/PROJECT_ID/face-mask-app
   ```

3. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy face-mask-app \
     --image gcr.io/PROJECT_ID/face-mask-app \
     --platform managed \
     --port 8501
   ```

---

## Production Considerations

### Model File
- The trained model (`models/mask_detector.h5`) is excluded from git due to size
- Options:
  1. Train on deployment server
  2. Use cloud storage (S3, GCS) and download on startup
  3. Use a smaller model or quantization

### Performance
- For production, consider using GPU instances
- Enable model caching in Streamlit
- Use load balancing for multiple users

### Security
- Add authentication if needed
- Use HTTPS in production
- Sanitize file uploads
- Set rate limiting

### Monitoring
- Add logging
- Monitor resource usage
- Track prediction accuracy
- Set up alerts
