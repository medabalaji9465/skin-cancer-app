# Skin Cancer Detection App

## Model file
Add your trained Keras model file as `model.h5` in the project root:

```text
skin-cancer-app/
├── app.py
├── database.py
├── model.h5
└── requirements.txt
```

The app expects an image input of `224x224x3` and a single prediction score where higher values indicate malignant.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Run with Docker
docker build -t skin-app .
docker run -p 8501:8501 skin-app
