import pandas as pd
import numpy as np
import warnings

# 1. Suppress warnings to keep the terminal perfectly clean
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_and_run_pipeline():
    print("⏳ Loading local medical dataset ('diabetes.csv')...")
    
    # Manually mapping the column names because the raw CSV should only contain numbers
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
    ]
    
    try:
        # Step 1: Read data, automatically trying to skip a text header row if it exists
        df = pd.read_csv('diabetes.csv', names=columns, skiprows=1)
        
        # Step 2: Robust Data Cleaning (Fixes the crash!)
        # Converts any non-numeric strings (like words) to NaN, then drops those rows.
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Explicitly make sure the target is an integer type for classification models
        df['Outcome'] = df['Outcome'].astype(int)
        
        print(f"✅ Successfully loaded and cleaned dataset! Final Data Shape: {df.shape}")
        
    except FileNotFoundError:
        print("\n❌ Error: 'diabetes.csv' was not found in your current directory!")
        print("💡 Make sure the file is renamed exactly to 'diabetes.csv' and placed in the same folder as this script.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 3. Separate Features (X) and Target Label (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 4. Train-Test Split (80% Training Data, 20% Testing Data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Feature Scaling (Now it works perfectly!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Initialize the 4 Classification Algorithms
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # 7. Train and Evaluate Each Model
    print("\n" + "="*50)
    print("         EVALUATING CLINICAL MODEL PERFORMANCES       ")
    print("="*50)
    
    summary_accuracies = {}

    for name, model in models.items():
        # Train model on scaled data
        model.fit(X_train_scaled, y_train)
        
        # Make disease predictions
        predictions = model.predict(X_test_scaled)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        summary_accuracies[name] = accuracy
        
        # Print detailed diagnostic classification reports
        print(f"\n🔹 Model: {name}")
        print(f"   Overall System Accuracy: {accuracy * 100:.2f}%")
        print("   Detailed Metrics Breakdown:")
        print(classification_report(y_test, predictions, target_names=["0 (Healthy)", "1 (Diabetes Positive)"]))
        
        # Print Confusion Matrix breakdown
        cm = confusion_matrix(y_test, predictions)
        print(f"   Confusion Matrix Layout:\n   [ True Healthy: {cm[0][0]:<3}  False Sick: {cm[0][1]} ]")
        print(f"   [ False Healthy: {cm[1][0]:<2}  True Sick:  {cm[1][1]} ]")
        print("-" * 50)

    # 8. Final Consolidated Summary Table
    print("\n" + "="*45)
    print("             FINAL SUMMARY SCORES            ")
    print("="*45)
    for name, acc in summary_accuracies.items():
        print(f" 🌟 {name:<25} : {acc * 100:.2f}%")
    print("="*45)

if __name__ == "__main__":
    load_and_run_pipeline()