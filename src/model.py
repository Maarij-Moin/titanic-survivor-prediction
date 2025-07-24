import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class TitanicSurvivorPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_data(self, filepath='data/titanic.csv'):
        """Load the Titanic dataset"""
        try:
            # Try local file first, then fallback to URL
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully from local file. Shape: {df.shape}")
        except FileNotFoundError:
            # Fallback to your GitHub URL
            url = "https://raw.githubusercontent.com/Maarij-Moin/titanic-survivor-prediction/refs/heads/main/data/Titanic_data.csv"
            df = pd.read_csv(url)
            print(f"Dataset loaded successfully from URL. Shape: {df.shape}")
        
        # Display basic info like in your notebook
        print("Dataset Information:")
        print(df.info())
        print("\nMissing values:")
        print(df.isnull().sum())
        
        return df
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        
        # Feature engineering 
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 60, 100], 
                                 labels=['Child', 'Teenager', 'Adult', 'Elderly'])
        data['FamilySize'] = data['SibSp'] + data['Parch']
        
        # Encode categorical variables 
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        
        # Drop unnecessary columns 
        columns_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
        data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)
        
        return data
    
    def prepare_features(self, data, target_column='Survived'):
        """Separate features and target, and prepare for training"""
        numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Sex']
        
        if target_column in data.columns:
            X = data[numeric_columns].copy()
            y = data[target_column]
        else:
            # For prediction on new data without target
            X = data[numeric_columns].copy() if all(col in data.columns for col in numeric_columns) else data
            y = None
        
        # Handle any remaining missing values
        X['Age'] = X['Age'].fillna(X['Age'].median())
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Apply standardization
        if y is not None:  # Training mode
            X_scaled = self.scaler.fit_transform(X)
        else:  # Prediction mode
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame to maintain column names
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and select the best one """
        # Split the data 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Support Vector Machine": SVC()
        }
        
        best_model = None
        best_accuracy = 0
        model_results = {}
        
        # Train and evaluate each model 
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            
            print(f"{name} Accuracy: {acc * 100:.2f}%")
            model_results[name] = acc
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
        
        # Store the best model
        self.model = best_model
        
        print(f"\nBest Model: {type(best_model).__name__}")
        print("Classification Report:")
        print(classification_report(y_test, best_model.predict(X_test)))
        
        return {
            'best_model': type(best_model).__name__,
            'best_accuracy': best_accuracy,
            'all_results': model_results,
            'X_test': X_test,
            'y_test': y_test,
            'predictions': best_model.predict(X_test)
        }
    
    def predict_single(self, passenger_data):
        """
        Predict survival for a single passenger
        
        Args:
            passenger_data (dict): Dictionary with passenger features
        Returns:
            int: 0 (Did not survive) or 1 (Survived)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame
        df = pd.DataFrame([passenger_data])
        
        # Apply same preprocessing
        df_processed = self.preprocess_data(df)
        X, _ = self.prepare_features(df_processed)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'probability_survived': float(probability[1]),
            'probability_not_survived': float(probability[0])
        }
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='models/titanic_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/titanic_model.pkl'):
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")

# Example usage function
def main():
    """
    Main function to train the model
    
    """
    # Initialize predictor
    predictor = TitanicSurvivorPredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    if df is not None:
        processed_data = predictor.preprocess_data(df)
        X, y = predictor.prepare_features(processed_data)
        
        # Train model
        results = predictor.train_model(X, y)
        
        # Show feature importance
        importance = predictor.get_feature_importance()
        if importance is not None:
            print("\nTop 10 Most Important Features:")
            print(importance.head(10))
        
        # Save model
        predictor.save_model()
        
        # Example prediction
        sample_passenger = {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 7.25,
            'Embarked': 'S'
        }
        
        try:
            prediction = predictor.predict_single(sample_passenger)
            print(f"\nSample Prediction: {prediction}")
        except Exception as e:
            print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()