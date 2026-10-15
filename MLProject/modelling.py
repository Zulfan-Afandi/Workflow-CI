import os
import sys
import argparse
import joblib
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def train(data_path="../namadataset_preprocessing.csv", n_estimators=100, max_depth=10):
    if not os.path.exists(data_path):
        fallbacks = [
            "../namadataset_preprocessing.csv",
            "namadataset_preprocessing.csv",
            "../../namadataset_preprocessing.csv"
        ]
        for p in fallbacks:
            if os.path.exists(p):
                data_path = p
                break
                
    print(f"[INFO] Memuat preprocessed dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    
    feature_cols = ['team', 'ban', 'role_ban', 'pick', 'role_pick']
    target_col = 'mvp'
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Data latih: {X_train.shape}, Data uji: {X_test.shape}")
    
    # Tracking lokal / CI
    tracking_dir = os.path.abspath("mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment("Mobile_Legends_M5_CI_Pipeline")
    
    with mlflow.start_run(run_name="CI_Run"):
        print(f"[INFO] Melatih RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth if max_depth and max_depth > 0 else None, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"[RESULT] Accuracy : {acc:.4f}")
        print(f"[RESULT] Precision: {prec:.4f}")
        print(f"[RESULT] Recall   : {rec:.4f}")
        print(f"[RESULT] F1-Score : {f1:.4f}")
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("[SUCCESS] Model dan metrik berhasil dicatat di MLflow!")
        
        joblib.dump(model, "model_ci.pkl")
        print("[SUCCESS] Model lokal disimpan di model_ci.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../namadataset_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train(args.data_path, args.n_estimators, args.max_depth)
