# 🌿 LeafLens AI – Plant Disease Detection

LeafLens AI is an AI-powered plant disease detection application that uses Deep Learning and Computer Vision to identify plant diseases from leaf images.

Built using a fine-tuned **ResNet50 Convolutional Neural Network (CNN)**, the application analyzes uploaded leaf images and predicts the corresponding disease class along with its confidence score.

## 🚀 Features

- Upload plant leaf images for disease detection.
- Predict plant diseases using a trained ResNet50 model.
- Identify **38 different plant disease classes**.
- Display prediction confidence scores.
- Show the top 5 predicted classes.
- Simple and interactive Streamlit interface.

## 🧠 Model Information

| Component | Details |
|---|---|
| Model Architecture | ResNet50 |
| Framework | TensorFlow / Keras |
| Learning Approach | Transfer Learning and Fine-Tuning |
| Input Image Size | 224 × 224 |
| Number of Classes | 38 |
| Output | Plant disease classification |

The model was trained using the PlantVillage dataset.

## 🛠️ Technology Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- ResNet50
- Jupyter Notebook

## 📁 Project Structure

```text
LeafLens-AI/
│
├── app.py
├── class_names.json
├── plant_disease_model.keras
├── model_training.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model is hosted separately in GitHub Releases to keep the source repository lightweight.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sarthakkk231/LeafLens-AI.git
```

### 2. Navigate to the project directory

```bash
cd LeafLens-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the trained model

Download the trained ResNet50 model from the link below:

**[Download Plant Disease Detection Model](https://github.com/Sarthakkk231/LeafLens-AI/releases/download/v1.0.0/plant_disease_model.keras)**

After downloading, place `plant_disease_model.keras` in the main project directory alongside `app.py`.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Training

The model training and evaluation workflow is documented in `model_training.ipynb`.

The notebook covers:

- Dataset preparation and preprocessing.
- Image resizing and augmentation.
- ResNet50 model initialization.
- Transfer learning.
- Fine-tuning.
- Model evaluation and performance visualization.
- Saving the trained model and class labels.

### Dataset

The project uses the PlantVillage dataset, which contains images of healthy and diseased plant leaves.

The dataset is downloaded separately for model training and is not included in this repository.

## ⚠️ Limitations

- Predictions depend on image quality and lighting conditions.
- The model is trained on PlantVillage images and may perform differently on real-world field images.
- Predictions should be treated as AI-assisted guidance, not as a substitute for expert agricultural diagnosis.
- The model recognizes only the 38 classes included in its training labels.

## 🔮 Future Improvements

- Improve performance on real-world field images.
- Add more plant species and disease categories.
- Integrate treatment and prevention recommendations.
- Develop a mobile-friendly interface.
- Deploy the application as a cloud-based service.

## 👨‍💻 Author

**Sarthak Karade**

AI & Data Science | Machine Learning | Deep Learning | AI Applications

## 📦 Model Download

The trained model is available under the repository's [Releases](https://github.com/Sarthakkk231/LeafLens-AI/releases) section.

---

⭐ If you find this project interesting, consider giving the repository a star!