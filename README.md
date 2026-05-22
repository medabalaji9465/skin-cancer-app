# Skin Cancer Detection App

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Run with Docker
docker build -t skin-app .
docker run -p 8501:8501 skin-app
