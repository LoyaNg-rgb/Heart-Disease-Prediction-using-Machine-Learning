# Heart Disease Prediction using Machine Learning

A comprehensive machine learning project for predicting heart disease using various classification algorithms with extensive data analysis, visualization, and model comparison.

## 🎯 Project Overview

This project implements and compares multiple machine learning models to predict heart disease based on patient health metrics. The pipeline includes data exploration, preprocessing, model training, hyperparameter tuning, and comprehensive evaluation.

## 📊 Dataset

The dataset contains patient health information with the following features:

- **Age**: Age of the patient
- **Sex**: Gender (M/F)
- **ChestPainType**: Type of chest pain (TA, ATA, NAP, ASY)
- **RestingBP**: Resting blood pressure
- **Cholesterol**: Serum cholesterol level
- **FastingBS**: Fasting blood sugar (1 if >120 mg/dl, 0 otherwise)
- **RestingECG**: Resting electrocardiogram results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise
- **ST_Slope**: Slope of the peak exercise ST segment
- **HeartDisease**: Target variable (1: heart disease, 0: no heart disease)

## 🚀 Features

- **Comprehensive Data Analysis**: Statistical summaries, distributions, and correlation analysis
- **Advanced Preprocessing**: Outlier handling, feature scaling, and categorical encoding
- **Multiple ML Models**: Logistic Regression, KNN, Random Forest, XGBoost
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Cross-Validation**: Robust model evaluation with stratified k-fold
- **Visualization**: Rich plots for data exploration and model comparison
- **Feature Importance**: Analysis of which features matter most
- **Model Deployment**: Ready-to-use prediction function

## 📁 Project Structure

```
heart-disease-prediction/
│
├── data/
│   └── heart.csv                 # Dataset
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── models.py                 # Model definitions and training
│   ├── evaluation.py             # Model evaluation and metrics
│   └── visualization.py          # Plotting functions
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── models/
│   ├── best_model.pkl           # Saved best model
│   ├── scaler.pkl               # Fitted scaler
│   └── feature_names.pkl        # Feature names
│
├── results/
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
│
├── requirements.txt             # Dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
├── LICENSE                     # License file
└── main.py                     # Main execution script
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

### Step-by-Step Analysis

```python
from src.heart_disease_predictor import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Load and explore data
df = predictor.load_and_explore_data("data/heart.csv")

# Run complete analysis
results = predictor.run_analysis()

# Make predictions
patient_data = {
    'Age': 54,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 150,
    'Cholesterol': 195,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 122,
    'ExerciseAngina': 'N',
    'Oldpeak': 0,
    'ST_Slope': 'Up'
}

risk_probability = predictor.predict_heart_disease(patient_data)
print(f"Heart Disease Risk: {risk_probability:.2%}")
```

## 🎯 Model Performance

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|------------|----------|-----------|---------|----------|
| Logistic Regression | 0.884 | 0.837 | 0.846 | 0.880 | 0.863 |
| K-Nearest Neighbors | 0.849 | 0.793 | 0.820 | 0.820 | 0.820 |
| Random Forest | 0.890 | 0.859 | 0.882 | 0.870 | 0.876 |
| **Random Forest (Tuned)** | **0.895** | **0.870** | **0.891** | **0.884** | **0.887** |
| XGBoost | 0.888 | 0.848 | 0.867 | 0.870 | 0.869 |
| XGBoost (Tuned) | 0.892 | 0.859 | 0.879 | 0.877 | 0.878 |

## 📊 Key Insights

### Top 5 Most Important Features:
1. **ST_Slope_Flat**: Flat ST slope during exercise
2. **ChestPainType_ASY**: Asymptomatic chest pain
3. **Oldpeak**: ST depression induced by exercise
4. **MaxHR**: Maximum heart rate achieved
5. **ExerciseAngina_Y**: Exercise-induced angina

### Model Insights:
- **Random Forest (Tuned)** achieved the best overall performance
- Cross-validation ROC-AUC scores range from 0.849 to 0.895
- All models show good generalization with minimal overfitting
- Feature importance analysis reveals exercise-related metrics as key predictors

## 🔬 Technical Details

- **Preprocessing**: MinMax scaling, outlier capping, one-hot encoding
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: Grid search with 3-fold CV
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Correlation heatmaps, ROC curves, confusion matrices

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Thanks to the open-source machine learning community
- Inspiration from various heart disease prediction research papers

## 📞 Contact

Your Name - Loyanganba Ngathem [loyanganba.ngathem@gmail.com]

Linkedin - www.linkedin.com/in/loyanganba-ngathem-315327378
---

⭐ If you found this project helpful, please give it a star!
