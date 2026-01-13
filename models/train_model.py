"""
CommissionIQ - ML Model for Predicting Commissioning Delays
Uses XGBoost for high-accuracy predictions with feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
from datetime import datetime
import os

class CommissionDelayPredictor:
    """Predict if commissioning activities will be delayed"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, df, is_training=True):
        """Feature engineering for commissioning data"""
        
        # Create copy
        data = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['equipment_type', 'system_type', 'activity_type', 
                          'complexity', 'contractor', 'resource_availability']
        
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                data[col + '_encoded'] = self.label_encoders[col].fit_transform(data[col])
            else:
                # Handle unseen categories
                data[col + '_encoded'] = data[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Convert boolean to int
        data['weather_sensitive_int'] = data['weather_sensitive'].astype(int)
        data['prior_similar_experience_int'] = data['prior_similar_experience'].astype(int)
        
        # Select features
        feature_cols = [
            'complexity_score',
            'planned_duration_days',
            'num_dependencies',
            'contractor_performance_score',
            'document_completeness_pct',
            'weather_sensitive_int',
            'prior_similar_experience_int',
            'equipment_type_encoded',
            'system_type_encoded',
            'activity_type_encoded',
            'complexity_encoded',
            'contractor_encoded',
            'resource_availability_encoded'
        ]
        
        X = data[feature_cols]
        
        if is_training:
            self.feature_names = feature_cols
            
        return X
    
    def train(self, activities_df):
        """Train the delay prediction model"""
        
        print("🔧 Preparing features...")
        X = self.prepare_features(activities_df, is_training=True)
        y = activities_df['is_delayed'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Delay rate: {y.mean()*100:.1f}%")
        
        # Train RandomForest model
        print("\n🚀 Training RandomForest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n📈 Model Performance:")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
        print(f"✅ ROC-AUC Score: {roc_auc:.3f}")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['On Time', 'Delayed']))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'test_size': len(X_test)
        }
    
    def predict(self, activities_df):
        """Predict delay probability for new activities"""
        
        X = self.prepare_features(activities_df, is_training=False)
        
        # Predict probabilities
        delay_proba = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        # Create results dataframe
        results = activities_df.copy()
        results['predicted_delay'] = predictions
        results['delay_probability'] = delay_proba
        results['risk_level'] = pd.cut(
            delay_proba, 
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
        
        return results
    
    def get_risk_factors(self, activity_data):
        """Identify top risk factors for a specific activity"""
        
        X = self.prepare_features(pd.DataFrame([activity_data]), is_training=False)
        
        # Get SHAP-like importance (simplified)
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            value = X.iloc[0, i]
            importance = self.feature_importance[
                self.feature_importance['feature'] == feature
            ]['importance'].values[0]
            feature_contributions[feature] = importance * abs(value)
        
        # Sort and return top factors
        sorted_factors = sorted(
            feature_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return sorted_factors
    
    def save_model(self, path=None):
        """Save trained model and encoders"""
        
        if path is None:
            # Default to models directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = current_dir + '/'
        
        with open(f'{path}delay_predictor.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{path}label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(f'{path}feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save feature importance
        self.feature_importance.to_csv(f'{path}feature_importance.csv', index=False)
        
        print(f"\n💾 Model saved to {path}")
    
    def load_model(self, path=None):
        """Load trained model"""
        
        if path is None:
            # Default to models directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = current_dir + '/'
        
        with open(f'{path}delay_predictor.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{path}label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open(f'{path}feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        self.feature_importance = pd.read_csv(f'{path}feature_importance.csv')
        
        print(f"✅ Model loaded from {path}")

def main():
    """Train and evaluate the model"""
    
    print("=" * 60)
    print("COMMISSIONIQ - DELAY PREDICTION MODEL")
    print("=" * 60)
    
    # Load data
    print("\n📁 Loading commissioning data...")
    
    # Get path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data', 'commissioning_activities.csv')
    
    activities_df = pd.read_csv(data_path)
    
    print(f"Loaded {len(activities_df)} commissioning activities")
    
    # Initialize predictor
    predictor = CommissionDelayPredictor()
    
    # Train model
    metrics = predictor.train(activities_df)
    
    # Save model
    predictor.save_model()
    
    # Test predictions on sample
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    sample = activities_df.sample(5)
    predictions = predictor.predict(sample)
    
    for idx, row in predictions.iterrows():
        print(f"\n🔍 Activity: {row['activity_id']}")
        print(f"   Equipment: {row['equipment_type']}")
        print(f"   Complexity: {row['complexity']}")
        print(f"   Actual Status: {'DELAYED' if row['is_delayed'] else 'ON TIME'}")
        print(f"   Predicted: {'DELAYED' if row['predicted_delay'] else 'ON TIME'}")
        print(f"   Delay Probability: {row['delay_probability']*100:.1f}%")
        print(f"   Risk Level: {row['risk_level']}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
