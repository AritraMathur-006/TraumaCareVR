import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

def train_and_export_model():
    print("--> Generating synthetic training dataset...")
    np.random.seed(42)
    n_samples = 1200

    data = {
        'pitch_jitter': np.random.uniform(0.01, 0.15, n_samples),
        'pause_ratio': np.random.uniform(0.05, 0.60, n_samples),
        'trauma_keywords': np.random.randint(0, 8, n_samples),
        'negative_sentiment': np.random.uniform(0.0, 1.0, n_samples),
        'days_to_hearing': np.random.randint(1, 90, n_samples),
        'intimidation_flag': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
    }

    df = pd.DataFrame(data)
    df['distress_score'] = np.clip(
        (df['pitch_jitter'] * 150) +
        (df['pause_ratio'] * 30) +
        (df['trauma_keywords'] * 5.0) +
        (df['negative_sentiment'] * 35.0) +
        (df['intimidation_flag'] * 25.0) +
        (np.maximum(0, (30 - df['days_to_hearing'])) * 0.4),
        0, 100
    )

    X = df.drop(columns=['distress_score'])
    y = df['distress_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("--> Fitting RandomForestRegressor...")
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    print(f"--> Training Complete. Model R² Score: {score:.4f}")

    joblib.dump(model, 'distress_model.joblib')
    print("--> Exported: 'distress_model.joblib'")

if __name__ == "__main__":
    train_and_export_model()
