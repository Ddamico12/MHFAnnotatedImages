import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load the dataset and preprocess it."""
    df = pd.read_csv('annotations.csv')
    return df

def analyze_correlations(df):
    """Analyze correlations between features and health status."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Analyze health correlations
    health_correlations = corr_matrix['fetal_health'].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    health_correlations.plot(kind='bar')
    plt.title('Feature Correlations with Health Status')
    plt.tight_layout()
    plt.savefig('health_correlations.png')
    plt.close()
    
    # Calculate mean values by health status
    health_means = df.groupby('fetal_health')[numeric_cols].mean()
    health_means.to_csv('health_means.csv')
    
    return corr_matrix, health_correlations, health_means

def main():
    # Load data
    df = load_data()
    
    # Analyze correlations
    corr_matrix, health_correlations, health_means = analyze_correlations(df)
    
    print("Analysis complete. Generated files:")
    print("- correlation_matrix.png")
    print("- health_correlations.png")
    print("- health_means.csv")

if __name__ == "__main__":
    main() 