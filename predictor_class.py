"""
Heart Disease Predictor - Main Class
===================================
Comprehensive machine learning pipeline for heart disease prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                   cross_val_score, StratifiedKFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


class HeartDiseasePredictor:
    """
    Comprehensive Heart Disease Prediction Pipeline
    
    This class encapsulates the entire machine learning pipeline for
    heart disease prediction including data preprocessing, model training,
    evaluation, and prediction capabilities.
    """
    
    def __init__(self):
        """Initialize the predictor with default configurations."""
        self.df = None
        self.df_encoded = None
        self.scaler = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.feature_names = None
        
        # Configuration
        self.categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        self.numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        self.random_state = 42
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for outputs."""
        directories = ['models', 'results', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def load_and_explore_data(self, filepath):
        """
        Load data and perform initial exploration.
        
        Args:
            filepath (str): Path to the CSV data file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("=== DATA OVERVIEW ===")
        self.df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget distribution:\n{self.df['HeartDisease'].value_counts()}")
        
        return self.df
    
    def explore_categorical_features(self):
        """Analyze categorical features and their relationship with target."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_explore_data first.")
        
        categorical_cols = self.categorical_cols + ['HeartDisease']
        
        # Distribution plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('Distribution of Categorical Variables', fontsize=16)
        
        for idx, col in enumerate(categorical_cols):
            if idx < 6:
                ax = axes[idx // 2, idx % 2]
                self.df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, X, y):
        """Plot confusion matrices for all models."""
        if not self.results:
            raise ValueError("No results available. Run model evaluation first.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        n_models = len(self.results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        fig.suptitle('Confusion Matrices', fontsize=16)
        
        if n_rows > 1:
            axes = axes.flatten()
        
        for idx, result in enumerate(self.results):
            model = result['model_obj']
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            
            ax = axes[idx] if n_rows > 1 else axes[idx % n_cols]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{result["model_name"]}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, n_rows * n_cols):
            if n_rows > 1:
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, model, model_name):
        """Analyze and plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'results/feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n=== TOP 10 FEATURES - {model_name} ===")
            for idx, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print(f"{model_name} doesn't have feature_importances_ attribute")
            return None
    
    def save_model_artifacts(self, best_model):
        """Save model artifacts for deployment."""
        print("\n=== SAVING MODEL ARTIFACTS ===")
        
        # Save model
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print("✅ Model artifacts saved:")
        print("   - models/best_model.pkl")
        print("   - models/scaler.pkl")
        print("   - models/feature_names.pkl")
    
    def load_model_artifacts(self):
        """Load saved model artifacts."""
        try:
            with open('models/best_model.pkl', 'rb') as f:
                self.best_model = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('models/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print("✅ Model artifacts loaded successfully")
            return True
        except FileNotFoundError:
            print("❌ Model artifacts not found. Train model first.")
            return False
    
    def predict_heart_disease(self, patient_data):
        """
        Predict heart disease for a new patient.
        
        Args:
            patient_data (dict): Patient features matching the original dataset
            
        Returns:
            float: Probability of heart disease, or None if prediction fails
        """
        if self.best_model is None or self.scaler is None or self.feature_names is None:
            if not self.load_model_artifacts():
                return None
        
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Apply preprocessing
            patient_encoded = pd.get_dummies(patient_df, columns=self.categorical_cols, drop_first=True)
            
            # Ensure all features are present
            for col in self.feature_names:
                if col not in patient_encoded.columns:
                    patient_encoded[col] = 0
            
            # Reorder columns to match training data
            patient_encoded = patient_encoded[self.feature_names]
            
            # Scale numerical features
            patient_encoded[self.numerical_cols] = self.scaler.transform(
                patient_encoded[self.numerical_cols]
            )
            
            # Make prediction
            prediction_proba = self.best_model.predict_proba(patient_encoded)[:, 1]
            return prediction_proba[0]
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def run_complete_analysis(self, filepath):
        """
        Run the complete analysis pipeline.
        
        Args:
            filepath (str): Path to the dataset
            
        Returns:
            dict: Analysis results and best model information
        """
        print("🚀 Starting Complete Heart Disease Prediction Analysis")
        print("=" * 60)
        
        # 1. Load and explore data
        print("\n📊 Step 1: Loading and exploring data...")
        self.load_and_explore_data(filepath)
        self.explore_categorical_features()
        
        # 2. Preprocess data
        print("\n🔧 Step 2: Preprocessing data...")
        self.preprocess_data()
        self.analyze_correlations()
        
        # 3. Prepare features and target
        X = self.df_encoded.drop("HeartDisease", axis=1)
        y = self.df_encoded["HeartDisease"]
        self.feature_names = list(X.columns)
        
        print(f"\nFinal feature set: {X.shape[1]} features")
        
        # 4. Setup and tune models
        print("\n🎯 Step 3: Setting up and tuning models...")
        self.setup_models()
        
        # Split for hyperparameter tuning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train)
        
        # 5. Evaluate all models
        print("\n📈 Step 4: Evaluating models...")
        print("=" * 50)
        
        self.results = []
        for name, model in self.models.items():
            result = self.evaluate_model_with_cv(model, X, y, name)
            self.results.append(result)
        
        # 6. Create visualizations
        print("\n📊 Step 5: Creating visualizations...")
        self.plot_model_comparison()
        self.plot_confusion_matrices(X, y)
        
        # 7. Analyze feature importance
        print("\n🔍 Step 6: Analyzing feature importance...")
        results_df = pd.DataFrame(self.results)
        best_model_idx = results_df['cv_mean'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'model_name']
        self.best_model = self.models[best_model_name]
        
        # Feature importance for tree-based models
        if 'Forest' in best_model_name or 'XGB' in best_model_name:
            self.analyze_feature_importance(self.best_model, best_model_name)
        
        # 8. Save model artifacts
        print("\n💾 Step 7: Saving model artifacts...")
        self.save_model_artifacts(self.best_model)
        
        # 9. Generate final summary
        results_summary = results_df[['model_name', 'cv_mean', 'cv_std', 'accuracy',
                                     'precision', 'recall', 'f1', 'roc_auc']]
        
        best_score = results_summary.loc[best_model_idx, 'cv_mean']
        
        print("\n" + "="*60)
        print("📋 ANALYSIS COMPLETE - FINAL SUMMARY")
        print("="*60)
        
        print("\n🏆 Best Model Performance:")
        print(f"   Model: {best_model_name}")
        print(f"   Cross-Validation ROC-AUC: {best_score:.4f}")
        
        print("\n📈 Performance Range:")
        print(f"   Best: {results_summary['cv_mean'].max():.4f}")
        print(f"   Worst: {results_summary['cv_mean'].min():.4f}")
        
        print(f"\n📊 Dataset Info:")
        print(f"   Samples: {len(self.df)}")
        print(f"   Features: {X.shape[1]} (after encoding)")
        print(f"   Class Distribution: {dict(y.value_counts())}")
        
        return {
            'summary': results_summary,
            'best_model_name': best_model_name,
            'best_score': best_score,
            'best_model': self.best_model,
            'feature_names': self.feature_names,
            'results': self.results
        }()
        plt.savefig('results/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature vs Target analysis
        categorical_features = [col for col in categorical_cols if col != 'HeartDisease']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distribution by Heart Disease Status', fontsize=16)
        
        for idx, col in enumerate(categorical_features):
            ax = axes[idx // 3, idx % 3]
            pd.crosstab(self.df[col], self.df['HeartDisease'], normalize='index').plot(
                kind='bar', ax=ax, color=['lightcoral', 'lightblue']
            )
            ax.set_title(f'{col} vs Heart Disease')
            ax.legend(['No Disease', 'Heart Disease'])
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/feature_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Complete preprocessing pipeline."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_explore_data first.")
        
        df_clean = self.df.copy()
        
        # Handle outliers
        outlier_cols = ['RestingBP', 'Cholesterol', 'Oldpeak']
        print("=== OUTLIER HANDLING ===")
        
        for col in outlier_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            
            outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            # Cap outliers using 99th percentile
            upper_cap = df_clean[col].quantile(0.99)
            lower_cap = df_clean[col].quantile(0.01)
            df_clean[col] = np.clip(df_clean[col], lower_cap, upper_cap)
            
            outliers_after = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            print(f"{col}: {outliers_before} outliers → {outliers_after} outliers")
        
        # Scale numerical features
        numerical_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        numerical_cols.remove('HeartDisease')  # Don't scale target
        
        self.scaler = MinMaxScaler()
        df_clean[numerical_cols] = self.scaler.fit_transform(df_clean[numerical_cols])
        
        # Encode categorical variables
        self.df_encoded = pd.get_dummies(df_clean, columns=self.categorical_cols, drop_first=True)
        
        print(f"\nFinal dataset shape: {self.df_encoded.shape}")
        return self.df_encoded
    
    def analyze_correlations(self):
        """Analyze feature correlations."""
        if self.df_encoded is None:
            raise ValueError("Data not preprocessed. Call preprocess_data first.")
        
        print("=== CORRELATION ANALYSIS ===")
        
        correlations = self.df_encoded.corr()
        target_corr = correlations['HeartDisease'].abs().sort_values(ascending=False)[1:]
        
        print("\nTop features by correlation with HeartDisease:")
        for feature, corr in target_corr.head(10).items():
            print(f"{feature:25}: {corr:.3f}")
        
        # Correlation heatmap
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlations, dtype=bool))
        sns.heatmap(correlations, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target_corr
    
    def setup_models(self):
        """Setup all models for training and evaluation."""
        # Basic models
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        }
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for selected models."""
        print("=== HYPERPARAMETER TUNING ===")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            rf_params, cv=3, scoring='roc_auc', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        # XGBoost tuning
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            xgb_params, cv=3, scoring='roc_auc', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        
        print(f"Best RF params: {rf_grid.best_params_}")
        print(f"Best RF score: {rf_grid.best_score_:.4f}")
        print(f"Best XGB params: {xgb_grid.best_params_}")
        print(f"Best XGB score: {xgb_grid.best_score_:.4f}")
        
        # Add tuned models
        self.models["Random Forest (Tuned)"] = rf_grid.best_estimator_
        self.models["XGBoost (Tuned)"] = xgb_grid.best_estimator_
        
        return rf_grid.best_estimator_, xgb_grid.best_estimator_
    
    def evaluate_model_with_cv(self, model, X, y, model_name, cv_folds=5):
        """Comprehensive model evaluation with cross-validation."""
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )
        
        # Train-test split for detailed metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'model_obj': model
        }
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        
        print(f"\n📊 {model_name}")
        print(f"CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision   : {metrics['precision']:.4f}")
        print(f"Recall      : {metrics['recall']:.4f}")
        print(f"F1 Score    : {metrics['f1']:.4f}")
        print(f"ROC AUC     : {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def plot_model_comparison(self):
        """Create comprehensive model comparison plots."""
        if not self.results:
            raise ValueError("No results available. Run model evaluation first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = [r['model_name'] for r in self.results]
        cv_means = [r['cv_mean'] for r in self.results]
        cv_stds = [r['cv_std'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        f1_scores = [r['f1'] for r in self.results]
        
        # CV ROC-AUC with error bars
        ax1.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        ax1.set_title('Cross-Validation ROC-AUC Scores')
        ax1.set_ylabel('ROC-AUC')
        ax1.tick_params(axis='x', rotation=45)
        
        # Test Accuracy
        ax2.bar(models, accuracies, alpha=0.7, color='orange')
        ax2.set_title('Test Set Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # F1 Scores
        ax3.bar(models, f1_scores, alpha=0.7, color='green')
        ax3.set_title('F1 Scores')
        ax3.set_ylabel('F1 Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # ROC Curves
        for result in self.results:
            if result['fpr'] is not None and result['tpr'] is not None:
                ax4.plot(result['fpr'], result['tpr'],
                        label=f"{result['model_name']} (AUC={result['roc_auc']:.3f})",
                        linewidth=2)
        
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout