import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXPERIMENT_NAME = "Spotify_Mode_Classification_Final"
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/mlflow", exist_ok=True)
os.makedirs("images/modeling", exist_ok=True)

# Configuration MLflow - stockage local
mlflow.set_tracking_uri("file:./data/mlflow")

def load_data():
    print("\nCHARGEMENT DES DONNEES")
    print("-"*40)
    
    X_train = pd.read_csv("data/processed/X_train_mode.csv")
    X_test = pd.read_csv("data/processed/X_test_mode.csv")
    y_train = pd.read_csv("data/processed/y_train_mode.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test_mode.csv").squeeze()
    
    print("X_train: {}".format(X_train.shape))
    print("X_test: {}".format(X_test.shape))
    print("y_train: {}".format(y_train.shape))
    print("y_test: {}".format(y_test.shape))
    
    print("\nDistribution de la target 'mode':")
    print("Mode 0 (mineur): {} train, {} test".format((y_train==0).sum(), (y_test==0).sum()))
    print("Mode 1 (majeur): {} train, {} test".format((y_train==1).sum(), (y_test==1).sum()))
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Mineur', 'Majeur'],
                yticklabels=['Mineur', 'Majeur'])
    plt.title('Matrice de Confusion: {}'.format(model_name))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    filename = "images/modeling/confusion_matrix_{}.png".format(model_name)
    plt.savefig(filename)
    plt.close()
    return filename

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: {}'.format(model_name))
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    filename = "images/modeling/roc_curve_{}.png".format(model_name)
    plt.savefig(filename)
    plt.close()
    return filename

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top {} Features - {}'.format(top_n, model_name))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        filename = "images/modeling/feature_importance_{}.png".format(model_name)
        plt.savefig(filename)
        plt.close()
        return filename
    return None

def run_experiment(model_name, model, param_grid, X_train, X_test, y_train, y_test, feature_names):
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=model_name):
        print("\n" + "="*50)
        print("Experimentation: {}".format(model_name))
        print("="*50)
        
        # Grid Search avec cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        # Entraînement
        grid_search.fit(X_train, y_train)
        
        # Meilleur modèle
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Prédictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Métriques
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        if y_test_proba is not None:
            auc = roc_auc_score(y_test, y_test_proba)
        
        # Log des paramètres et métriques dans MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        if y_test_proba is not None:
            mlflow.log_metric("auc", auc)
        
        # Affichage des résultats
        print("\nBest Parameters: {}".format(best_params))
        print("Train Accuracy: {:.2%}".format(train_acc))
        print("Test Accuracy: {:.2%}".format(test_acc))
        print("Train F1: {:.4f}".format(train_f1))
        print("Test F1: {:.4f}".format(test_f1))
        if y_test_proba is not None:
            print("AUC: {:.3f}".format(auc))
        
        # Sauvegarde des graphiques
        cm_path = plot_confusion_matrix(y_test, y_test_pred, model_name)
        mlflow.log_artifact(cm_path)
        print("Confusion matrix saved: {}".format(cm_path))
        
        if y_test_proba is not None:
            roc_path = plot_roc_curve(y_test, y_test_proba, model_name)
            mlflow.log_artifact(roc_path)
            print("ROC curve saved: {}".format(roc_path))
        
        # Feature importance
        fi_path = plot_feature_importance(best_model, feature_names, model_name)
        if fi_path:
            mlflow.log_artifact(fi_path)
            print("Feature importance saved: {}".format(fi_path))
        
        # Log du modèle
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, test_f1

if __name__ == "__main__":
    print("="*60)
    print("SPOTIFY MODE CLASSIFICATION")
    print("="*60)
    
    # 1. Charger les données
    X_train, X_test, y_train, y_test = load_data()
    feature_names = X_train.columns.tolist()
    
    # 2. Baseline Dummy (optionnel)
    print("\n" + "="*50)
    print("BASELINE - Dummy Classifier")
    print("="*50)
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
    print("Dummy Accuracy: {:.2%}".format(dummy_acc))
    
    # 3. Configuration des modèles (minimum 2 comme demandé)
    models_config = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, 
                n_jobs=-1, 
                use_label_encoder=False, 
                eval_metric='logloss'
            ),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.05, 0.1]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
            "params": {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs']
            }
        }
    }
    
    # 4. Lancer les expériences
    results = {}
    best_f1 = -1
    best_model = None
    best_name = ""
    
    for name, config in models_config.items():
        model, f1 = run_experiment(
            name, 
            config["model"], 
            config["params"],
            X_train, X_test, 
            y_train, y_test,
            feature_names
        )
        results[name] = f1
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
    
    # 5. Comparaison finale
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, f1 in results.items():
        print("{}: F1 = {:.4f}".format(name, f1))
    
    print("\nBest Model: {} with F1 = {:.4f}".format(best_name, best_f1))
    
    # 6. Sauvegarde du meilleur modèle
    scaler = joblib.load("data/models/scaler_mode.pkl")
    
    pipeline = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_f1': best_f1,
        'best_name': best_name
    }
    
    joblib.dump(pipeline, "data/models/best_mode_pipeline.pkl")
    print("\nBest model saved: data/models/best_mode_pipeline.pkl")
    
    # 7. Instructions MLflow
    print("\n" + "="*60)
    print("MLFLOW UI")
    print("="*60)
    print("To view results in MLflow, run:")
    print("  mlflow ui --backend-store-uri file:./data/mlflow")
    print("Then open http://localhost:5000 in your browser")
    print("="*60)