"""
Heart Disease Prediction - Main Execution Script
===============================================
A comprehensive machine learning pipeline for heart disease prediction.

Author: Your Name
Date: 2025
"""

import os
import sys
import warnings
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from heart_disease_predictor import HeartDiseasePredictor

warnings.filterwarnings('ignore')


def main():
    """
    Main execution function for the heart disease prediction pipeline.
    """
    print("🫀 Heart Disease Prediction Pipeline")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Check if data file exists
    data_path = "data/heart.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure the heart.csv file is in the data/ directory")
        return
    
    try:
        # Run the complete analysis
        print("\n🚀 Starting comprehensive analysis...")
        results = predictor.run_complete_analysis(data_path)
        
        print("\n✅ Analysis completed successfully!")
        print("\n📊 Results Summary:")
        print(results['summary'].round(4))
        
        print(f"\n🏆 Best Model: {results['best_model_name']}")
        print(f"   Cross-Validation ROC-AUC: {results['best_score']:.4f}")
        
        # Example prediction
        print("\n🔮 Example Prediction:")
        example_patient = {
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
        
        risk_prob = predictor.predict_heart_disease(example_patient)
        if risk_prob is not None:
            print(f"   Patient Risk Probability: {risk_prob:.2%}")
            risk_level = "High" if risk_prob > 0.5 else "Low"
            print(f"   Risk Level: {risk_level}")
        
        print("\n📁 Model artifacts saved in models/ directory")
        print("📈 Visualizations displayed during analysis")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure all required files are in their correct locations")
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        print("Please check the data format and try again")


if __name__ == "__main__":
    main()