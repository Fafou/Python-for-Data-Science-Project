import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_advanced_eda(file_path="data/raw/spotify_songs.csv"):
    """
    Performs Advanced Exploratory Data Analysis on the Spotify dataset.
    Adapted from the professor's bank churn example.
    """
    # MODIFICATION 1: Ajout de low_memory=False
    df = pd.read_csv(file_path, low_memory=False)
    
    # MODIFICATION 2: Forcer les colonnes numériques
    numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("--- Dataset Shape ---")
    print(df.shape)
    
    # 1. Target Imbalance Analysis
    plt.figure(figsize=(12, 6))
    df['genre'].value_counts(normalize=True).head(15).plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Genre Proportion (Top 15)')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/eda/target_imbalance.png')
    plt.close()
    
    # 2. Categorical Analysis
    categorical_features = [col for col in ['key', 'mode'] if col in df.columns]
    if categorical_features:
        fig, axes = plt.subplots(1, len(categorical_features), figsize=(12, 5))
        if len(categorical_features) == 1:
            axes = [axes]
        for i, feature in enumerate(categorical_features):
            sns.countplot(x=feature, hue='genre', data=df.sample(min(5000, len(df))), ax=axes[i], palette='muted')
            axes[i].set_title(f'Genre by {feature}')
            axes[i].legend(title='Genre', loc='upper right')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig('images/eda/categorical_impact.png')
        plt.close()
    
    # 3. Numerical Analysis: Outlier Detection
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features[:6]):
        plt.subplot(2, 3, i+1)
        sns.boxplot(y=df[feature], palette='Set2')
        plt.title(f'{feature} Distribution')
    plt.tight_layout()
    plt.savefig('images/eda/numerical_boxplots.png')
    plt.close()
    
    # 4. Multivariate Analysis: Pairplot
    sample_cols = numerical_features[:4] + ['genre']
    sample_df = df[sample_cols].dropna().sample(min(1000, len(df)))
    sns.pairplot(sample_df, hue='genre', diag_kind='kde', palette='husl')
    plt.savefig('images/eda/multivariate_pairplot.png')
    plt.close()
    
    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('images/eda/correlation_heatmap.png')
    plt.close()

    print("\n✅ Advanced EDA completed. Plots saved in 'images/eda/' directory.")
    
    # Bonus: quelques stats
    print("\n📊 Top 10 genres:")
    print(df['genre'].value_counts().head(10))

if __name__ == "__main__":
    run_advanced_eda()