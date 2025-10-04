#!/usr/bin/env python
# coding: utf-8

"""
AI-Powered Personalized Exam Preparation Assistant for Competitive Exams
VISUAL Machine Learning Laboratory Experiments (1-9)
Focus: Beautiful Graphs and Visualizations
"""

import os
import re
import json
import time
import random
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# Import our existing TNPSC Assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS, load_questions, load_syllabus

# Set style for beautiful plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

print("="*60)
print("🎨 AI-POWERED TNPSC EXAM PREPARATION ASSISTANT")
print("🎯 VISUAL MACHINE LEARNING EXPERIMENTS")
print("📊 FOCUS: BEAUTIFUL GRAPHS & VISUALIZATIONS")
print("="*60)

class VisualTNPSCMLExperiments:
    """Class containing all 9 ML experiments with focus on beautiful visualizations"""
    
    def __init__(self):
        self.assistant = TNPSCExamAssistant()
        self.questions_df = self.assistant.questions_df
        self.syllabus = self.assistant.syllabus
        print("🚀 Visual TNPSC ML Experiments initialized successfully!")
        
        # Set up matplotlib for better plots
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def experiment_1_dataset_visualization(self):
        """
        EXPERIMENT 1: Dataset Loading with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("📊 EXPERIMENT 1: DATASET VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TNPSC Dataset Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Questions per Unit (Bar Chart)
        unit_counts = df['unit'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unit_counts)))
        bars = axes[0,0].bar(unit_counts.index, unit_counts.values, color=colors)
        axes[0,0].set_title('Questions per Unit', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Unit Number')
        axes[0,0].set_ylabel('Number of Questions')
        
        # Add value labels on bars
        for bar, value in zip(bars, unit_counts.values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                          str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Difficulty Distribution (Pie Chart)
        difficulty_counts = df['difficulty'].value_counts()
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
        wedges, texts, autotexts = axes[0,1].pie(difficulty_counts.values, 
                                                labels=difficulty_counts.index,
                                                autopct='%1.1f%%',
                                                colors=colors_pie,
                                                explode=(0.05, 0.05, 0.05),
                                                shadow=True,
                                                startangle=90)
        axes[0,1].set_title('Difficulty Distribution', fontsize=14, fontweight='bold')
        
        # 3. Year-wise Distribution (Line Plot)
        year_counts = df['year'].value_counts().sort_index()
        axes[0,2].plot(year_counts.index, year_counts.values, 'o-', linewidth=3, markersize=8, color='#e74c3c')
        axes[0,2].fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#e74c3c')
        axes[0,2].set_title('Questions by Year', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Year')
        axes[0,2].set_ylabel('Number of Questions')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Unit vs Difficulty Heatmap
        unit_difficulty = pd.crosstab(df['unit'], df['difficulty'])
        sns.heatmap(unit_difficulty, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=axes[1,0], cbar_kws={'label': 'Count'})
        axes[1,0].set_title('Unit vs Difficulty Heatmap', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Difficulty Level')
        axes[1,0].set_ylabel('Unit Number')
        
        # 5. Question Length Distribution
        df['question_length'] = df['question'].str.len()
        axes[1,1].hist(df['question_length'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1,1].axvline(df['question_length'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["question_length"].mean():.1f}')
        axes[1,1].set_title('Question Length Distribution', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Question Length (characters)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        # 6. Unit Names with Counts (Horizontal Bar)
        unit_names = [f"Unit {i}\n{UNITS[i]}" for i in unit_counts.index]
        y_pos = np.arange(len(unit_names))
        bars = axes[1,2].barh(y_pos, unit_counts.values, color=colors)
        axes[1,2].set_yticks(y_pos)
        axes[1,2].set_yticklabels(unit_names, fontsize=10)
        axes[1,2].set_title('Units Overview', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Number of Questions')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, unit_counts.values)):
            axes[1,2].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                          str(value), ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ RESULT: Beautiful dataset visualizations created successfully!")
        return df
    
    def experiment_2_statistical_analysis(self):
        """
        EXPERIMENT 2: Advanced Statistical Analysis with Visualizations
        """
        print("\n" + "="*50)
        print("📈 EXPERIMENT 2: STATISTICAL ANALYSIS")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Create statistical dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('TNPSC Statistical Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Box Plot - Question Length by Difficulty
        df['question_length'] = df['question'].str.len()
        sns.boxplot(data=df, x='difficulty', y='question_length', ax=axes[0,0])
        axes[0,0].set_title('Question Length by Difficulty', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Difficulty Level')
        axes[0,0].set_ylabel('Question Length')
        
        # 2. Violin Plot - Distribution Analysis
        sns.violinplot(data=df, x='unit', y='question_length', ax=axes[0,1])
        axes[0,1].set_title('Question Length Distribution by Unit', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Unit')
        axes[0,1].set_ylabel('Question Length')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Correlation Matrix
        # Create numerical features for correlation
        df['difficulty_num'] = df['difficulty'].map({'easy': 1, 'medium': 2, 'hard': 3})
        df['word_count'] = df['question'].str.split().str.len()
        
        corr_data = df[['unit', 'year', 'question_length', 'word_count', 'difficulty_num']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0,2])
        axes[0,2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 4. Stacked Bar Chart - Unit vs Year vs Difficulty
        pivot_data = df.pivot_table(index='unit', columns='difficulty', values='year', aggfunc='count', fill_value=0)
        pivot_data.plot(kind='bar', stacked=True, ax=axes[1,0], color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1,0].set_title('Questions by Unit and Difficulty', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Unit')
        axes[1,0].set_ylabel('Number of Questions')
        axes[1,0].legend(title='Difficulty')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 5. Scatter Plot with Regression Line
        sns.scatterplot(data=df, x='question_length', y='word_count', hue='difficulty', 
                       size='unit', sizes=(50, 200), ax=axes[1,1])
        sns.regplot(data=df, x='question_length', y='word_count', scatter=False, 
                   color='red', ax=axes[1,1])
        axes[1,1].set_title('Question Length vs Word Count', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Question Length (characters)')
        axes[1,1].set_ylabel('Word Count')
        
        # 6. Radar Chart for Unit Statistics
        from math import pi
        
        # Calculate statistics per unit
        unit_stats = df.groupby('unit').agg({
            'question_length': 'mean',
            'word_count': 'mean',
            'difficulty_num': 'mean',
            'year': 'mean'
        }).round(2)
        
        # Normalize for radar chart
        unit_stats_norm = (unit_stats - unit_stats.min()) / (unit_stats.max() - unit_stats.min())
        
        # Create radar chart for first 4 units
        categories = list(unit_stats_norm.columns)
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        axes[1,2].set_theta_offset(pi / 2)
        axes[1,2].set_theta_direction(-1)
        axes[1,2].set_thetagrids(np.degrees(angles[:-1]), categories)
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, unit in enumerate(unit_stats_norm.index[:4]):
            values = unit_stats_norm.loc[unit].values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            axes[1,2].plot(angles, values, 'o-', linewidth=2, label=f'Unit {unit}', color=colors[i])
            axes[1,2].fill(angles, values, alpha=0.25, color=colors[i])
        
        axes[1,2].set_title('Unit Characteristics Radar', fontsize=14, fontweight='bold')
        axes[1,2].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ RESULT: Advanced statistical visualizations created successfully!")
        return unit_stats
    
    def experiment_3_linear_regression_visual(self):
        """
        EXPERIMENT 3: Linear Regression with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("📊 EXPERIMENT 3: LINEAR REGRESSION VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Prepare features
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        df['difficulty_score'] = df['difficulty'].map(difficulty_map)
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['avg_option_length'] = (df['option_a'].str.len() + df['option_b'].str.len() + 
                                  df['option_c'].str.len() + df['option_d'].str.len()) / 4
        
        features = ['unit', 'year', 'question_length', 'word_count', 'avg_option_length']
        X = df[features]
        y = df['difficulty_score']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create visualization dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter Plot
        axes[0,0].scatter(y_test, y_pred, alpha=0.7, color='blue', s=100, edgecolors='black')
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        axes[0,0].set_xlabel('Actual Difficulty Score')
        axes[0,0].set_ylabel('Predicted Difficulty Score')
        axes[0,0].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        axes[0,0].text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', 
                      transform=axes[0,0].transAxes, fontsize=12, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 2. Residuals Plot
        residuals = y_test - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.7, color='green', s=100, edgecolors='black')
        axes[0,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('Predicted Difficulty Score')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        
        # 3. Feature Importance (Coefficients)
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=True)
        
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
        bars = axes[0,2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        axes[0,2].set_xlabel('Coefficient Value')
        axes[0,2].set_title('Feature Importance (Coefficients)', fontsize=14, fontweight='bold')
        axes[0,2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, coef_df['Coefficient']):
            axes[0,2].text(value + (0.01 if value >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                          f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center')
        
        # 4. Prediction Error Distribution
        axes[1,0].hist(residuals, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Mean: {residuals.mean():.3f}')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1,0].legend()
        
        # 5. Learning Curve Simulation
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            n_samples = int(size * len(X_train))
            if n_samples < 2:
                n_samples = 2
            
            X_subset = X_train.iloc[:n_samples]
            y_subset = y_train.iloc[:n_samples]
            
            temp_model = LinearRegression()
            temp_model.fit(X_subset, y_subset)
            
            train_pred = temp_model.predict(X_subset)
            test_pred = temp_model.predict(X_test)
            
            train_scores.append(r2_score(y_subset, train_pred))
            test_scores.append(r2_score(y_test, test_pred))
        
        axes[1,1].plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
        axes[1,1].plot(train_sizes, test_scores, 'o-', color='red', label='Validation Score')
        axes[1,1].set_xlabel('Training Set Size')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title('Learning Curve', fontsize=14, fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Feature vs Target Relationships
        # Show relationship between most important feature and target
        most_important_feature = coef_df.iloc[-1]['Feature']
        axes[1,2].scatter(df[most_important_feature], df['difficulty_score'], 
                         alpha=0.7, color='orange', s=100, edgecolors='black')
        
        # Add trend line
        z = np.polyfit(df[most_important_feature], df['difficulty_score'], 1)
        p = np.poly1d(z)
        axes[1,2].plot(df[most_important_feature], p(df[most_important_feature]), "r--", alpha=0.8)
        
        axes[1,2].set_xlabel(most_important_feature)
        axes[1,2].set_ylabel('Difficulty Score')
        axes[1,2].set_title(f'{most_important_feature} vs Difficulty', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Model Performance:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Mean Squared Error: {mse:.4f}")
        print("✅ RESULT: Linear regression visualizations created successfully!")
        
        return model, r2, mse
    
    def experiment_4_classification_visual(self):
        """
        EXPERIMENT 4: Classification Models with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("🎯 EXPERIMENT 4: CLASSIFICATION VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Prepare features for classification
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        df['question_length'] = df['question'].str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        X_numerical = df[['unit', 'year', 'question_length', 'has_numbers']].values
        
        X = np.hstack([X_text, X_numerical])
        y = df['difficulty']  # Multi-class classification
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train models
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_proba = svm_model.predict_proba(X_test)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        
        # Create visualization dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classification Models Comparison Dashboard', fontsize=20, fontweight='bold')
        
        # 1. SVM Confusion Matrix
        cm_svm = confusion_matrix(y_test, svm_pred)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['easy', 'hard', 'medium'], 
                   yticklabels=['easy', 'hard', 'medium'], ax=axes[0,0])
        axes[0,0].set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Random Forest Confusion Matrix
        cm_rf = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['easy', 'hard', 'medium'], 
                   yticklabels=['easy', 'hard', 'medium'], ax=axes[0,1])
        axes[0,1].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. Model Accuracy Comparison
        svm_accuracy = accuracy_score(y_test, svm_pred)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        models = ['SVM', 'Random Forest']
        accuracies = [svm_accuracy, rf_accuracy]
        colors = ['#3498db', '#2ecc71']
        
        bars = axes[0,2].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        axes[0,2].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0,2].set_ylabel('Accuracy')
        axes[0,2].set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prediction Probability Distribution (SVM)
        prob_df = pd.DataFrame(svm_proba, columns=['easy', 'hard', 'medium'])
        prob_df.plot(kind='hist', alpha=0.7, bins=15, ax=axes[1,0])
        axes[1,0].set_title('SVM Prediction Probabilities', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # 5. Feature Importance (Random Forest)
        feature_names = (list(vectorizer.get_feature_names_out()) + 
                        ['unit', 'year', 'question_length', 'has_numbers'])
        
        # Get top 10 most important features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        bars = axes[1,1].barh(range(len(importance_df)), importance_df['importance'], 
                             color='orange', alpha=0.7)
        axes[1,1].set_yticks(range(len(importance_df)))
        axes[1,1].set_yticklabels(importance_df['feature'])
        axes[1,1].set_title('Top 10 Feature Importance (RF)', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Importance')
        
        # 6. Classification Report Visualization
        from sklearn.metrics import precision_recall_fscore_support
        
        # Get metrics for both models
        svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(y_test, svm_pred, average=None)
        rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average=None)
        
        classes = ['easy', 'hard', 'medium']
        x = np.arange(len(classes))
        width = 0.15
        
        axes[1,2].bar(x - width*1.5, svm_precision, width, label='SVM Precision', alpha=0.8)
        axes[1,2].bar(x - width/2, svm_recall, width, label='SVM Recall', alpha=0.8)
        axes[1,2].bar(x + width/2, rf_precision, width, label='RF Precision', alpha=0.8)
        axes[1,2].bar(x + width*1.5, rf_recall, width, label='RF Recall', alpha=0.8)
        
        axes[1,2].set_xlabel('Classes')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_title('Precision & Recall Comparison', fontsize=14, fontweight='bold')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(classes)
        axes[1,2].legend()
        axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Model Performance:")
        print(f"   SVM Accuracy: {svm_accuracy:.4f}")
        print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")
        print("✅ RESULT: Classification visualizations created successfully!")
        
        return svm_model, rf_model, svm_accuracy, rf_accuracy  
  
    def experiment_5_clustering_visual(self):
        """
        EXPERIMENT 5: Clustering Analysis with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("🎨 EXPERIMENT 5: CLUSTERING VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Prepare features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        df['question_length'] = df['question'].str.len()
        X_numerical = df[['question_length', 'year']].values
        
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        X = np.hstack([X_text, X_numerical_scaled])
        
        # Apply different clustering methods
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm_labels = gmm.fit_predict(X)
        gmm_proba = gmm.predict_proba(X)
        
        # Hierarchical clustering (on subset for performance)
        subset_size = min(30, len(X))
        X_subset = X[:subset_size]
        linkage_matrix = linkage(X_subset, method='ward')
        hierarchical_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        X_subset_pca = X_pca[:subset_size]
        
        # Create clustering visualization dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Clustering Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. K-Means Clustering
        scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                                   cmap='viridis', alpha=0.7, s=100, edgecolors='black')
        
        # Plot cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        axes[0,0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', 
                         marker='x', s=300, linewidths=3, label='Centroids')
        
        axes[0,0].set_title('K-Means Clustering', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('First Principal Component')
        axes[0,0].set_ylabel('Second Principal Component')
        axes[0,0].legend()
        plt.colorbar(scatter, ax=axes[0,0], label='Cluster')
        
        # 2. Gaussian Mixture Model
        scatter = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, 
                                   cmap='plasma', alpha=0.7, s=100, edgecolors='black')
        axes[0,1].set_title('Gaussian Mixture Model', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('First Principal Component')
        axes[0,1].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[0,1], label='Cluster')
        
        # 3. Hierarchical Clustering with Dendrogram
        dendrogram(linkage_matrix, ax=axes[0,2], truncate_mode='level', p=3)
        axes[0,2].set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Sample Index')
        axes[0,2].set_ylabel('Distance')
        
        # 4. Actual Unit Labels (Ground Truth)
        scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['unit'], 
                                   cmap='tab10', alpha=0.7, s=100, edgecolors='black')
        axes[1,0].set_title('Actual Unit Labels (Ground Truth)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('First Principal Component')
        axes[1,0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[1,0], label='Unit')
        
        # 5. GMM Probability Confidence
        max_proba = np.max(gmm_proba, axis=1)
        scatter = axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=max_proba, 
                                   cmap='coolwarm', alpha=0.7, s=100, edgecolors='black')
        axes[1,1].set_title('GMM Assignment Confidence', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('First Principal Component')
        axes[1,1].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[1,1], label='Max Probability')
        
        # 6. Clustering Comparison Metrics
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        
        # Calculate metrics
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        gmm_silhouette = silhouette_score(X, gmm_labels)
        
        # Compare with ground truth (units)
        kmeans_ari = adjusted_rand_score(df['unit'], kmeans_labels)
        gmm_ari = adjusted_rand_score(df['unit'], gmm_labels)
        
        methods = ['K-Means', 'GMM']
        silhouette_scores = [kmeans_silhouette, gmm_silhouette]
        ari_scores = [kmeans_ari, gmm_ari]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = axes[1,2].bar(x - width/2, silhouette_scores, width, 
                             label='Silhouette Score', alpha=0.8, color='skyblue')
        bars2 = axes[1,2].bar(x + width/2, ari_scores, width, 
                             label='Adjusted Rand Index', alpha=0.8, color='lightcoral')
        
        axes[1,2].set_xlabel('Clustering Method')
        axes[1,2].set_ylabel('Score')
        axes[1,2].set_title('Clustering Performance Metrics', fontsize=14, fontweight='bold')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(methods)
        axes[1,2].legend()
        axes[1,2].set_ylim(-0.5, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                              f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Clustering Performance:")
        print(f"   K-Means Silhouette Score: {kmeans_silhouette:.4f}")
        print(f"   GMM Silhouette Score: {gmm_silhouette:.4f}")
        print(f"   K-Means ARI with Units: {kmeans_ari:.4f}")
        print(f"   GMM ARI with Units: {gmm_ari:.4f}")
        print("✅ RESULT: Clustering visualizations created successfully!")
        
        return kmeans, gmm, kmeans_silhouette, gmm_silhouette
    
    def experiment_6_pca_visual(self):
        """
        EXPERIMENT 6: PCA Analysis with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("🔍 EXPERIMENT 6: PCA VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Prepare comprehensive feature set
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Additional features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        
        X_numerical = df[['unit', 'year', 'question_length', 'word_count', 
                         'has_numbers', 'has_punctuation', 'avg_word_length']].values
        
        # Combine features
        X = np.hstack([X_text, X_numerical])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Create PCA visualization dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Principal Component Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Scree Plot
        n_components = min(20, len(explained_variance_ratio))
        axes[0,0].plot(range(1, n_components + 1), explained_variance_ratio[:n_components], 
                      'bo-', linewidth=2, markersize=8)
        axes[0,0].set_title('Scree Plot', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Principal Component')
        axes[0,0].set_ylabel('Explained Variance Ratio')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add elbow point
        elbow_point = 5  # Typically around 5 components
        axes[0,0].axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7, label='Elbow Point')
        axes[0,0].legend()
        
        # 2. Cumulative Explained Variance
        axes[0,1].plot(range(1, n_components + 1), cumulative_variance[:n_components], 
                      'ro-', linewidth=2, markersize=8)
        axes[0,1].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Variance')
        axes[0,1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
        axes[0,1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Principal Component')
        axes[0,1].set_ylabel('Cumulative Variance Ratio')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 2D PCA - Colored by Unit
        scatter = axes[0,2].scatter(X_pca[:, 0], X_pca[:, 1], c=df['unit'], 
                                   cmap='tab10', alpha=0.7, s=100, edgecolors='black')
        axes[0,2].set_title('PCA: Questions by Unit', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
        axes[0,2].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[0,2], label='Unit')
        
        # 4. 2D PCA - Colored by Difficulty
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        difficulty_numeric = df['difficulty'].map(difficulty_map)
        scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=difficulty_numeric, 
                                   cmap='viridis', alpha=0.7, s=100, edgecolors='black')
        axes[1,0].set_title('PCA: Questions by Difficulty', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
        axes[1,0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
        cbar = plt.colorbar(scatter, ax=axes[1,0])
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['Easy', 'Medium', 'Hard'])
        
        # 5. 2D PCA - Colored by Year
        scatter = axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=df['year'], 
                                   cmap='plasma', alpha=0.7, s=100, edgecolors='black')
        axes[1,1].set_title('PCA: Questions by Year', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
        axes[1,1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
        plt.colorbar(scatter, ax=axes[1,1], label='Year')
        
        # 6. Component Loadings (Feature Contributions)
        feature_names = (list(vectorizer.get_feature_names_out()) + 
                        ['unit', 'year', 'question_length', 'word_count', 
                         'has_numbers', 'has_punctuation', 'avg_word_length'])
        
        # Get loadings for PC1 and PC2
        pc1_loadings = pca.components_[0]
        pc2_loadings = pca.components_[1]
        
        # Show top contributing features
        top_features_pc1 = np.argsort(np.abs(pc1_loadings))[-10:]
        top_features_pc2 = np.argsort(np.abs(pc2_loadings))[-10:]
        
        # Create loading plot for PC1
        y_pos = np.arange(len(top_features_pc1))
        axes[1,2].barh(y_pos, pc1_loadings[top_features_pc1], alpha=0.7, color='blue')
        axes[1,2].set_yticks(y_pos)
        axes[1,2].set_yticklabels([feature_names[i] for i in top_features_pc1])
        axes[1,2].set_title('Top Features Contributing to PC1', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Loading Value')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"📊 PCA Analysis Summary:")
        print(f"   Original dimensions: {X.shape[1]}")
        print(f"   Components for 95% variance: {n_components_95}")
        print(f"   PC1 explains {explained_variance_ratio[0]:.1%} of variance")
        print(f"   PC2 explains {explained_variance_ratio[1]:.1%} of variance")
        print("✅ RESULT: PCA visualizations created successfully!")
        
        return pca, X_pca, explained_variance_ratio
    
    def experiment_8_decision_tree_visual(self):
        """
        EXPERIMENT 8: Decision Tree with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("🌳 EXPERIMENT 8: DECISION TREE VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Create comprehensive features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        df['question_mark_count'] = df['question'].str.count(r'\?')
        df['capital_letters_count'] = df['question'].str.count(r'[A-Z]')
        
        # Keyword-based features
        science_keywords = ['physics', 'chemistry', 'biology', 'science', 'atom', 'cell', 'energy']
        history_keywords = ['history', 'ancient', 'medieval', 'empire', 'dynasty', 'culture']
        geography_keywords = ['geography', 'river', 'mountain', 'climate', 'ocean', 'continent']
        
        df['science_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in science_keywords if keyword in x))
        df['history_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in history_keywords if keyword in x))
        df['geography_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in geography_keywords if keyword in x))
        
        # Features and target
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers', 
                          'has_punctuation', 'avg_word_length', 'question_mark_count',
                          'capital_letters_count', 'science_keywords', 'history_keywords',
                          'geography_keywords']
        
        X = df[feature_columns]
        y = df['unit']  # Predict unit
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train decision tree
        dt_classifier = DecisionTreeClassifier(
            criterion='gini',
            max_depth=4,  # Reduced for better visualization
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        dt_classifier.fit(X_train, y_train)
        y_pred = dt_classifier.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create decision tree visualization dashboard
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Decision Tree Visualization (Large plot)
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
        plot_tree(dt_classifier, 
                 feature_names=feature_columns,
                 class_names=[f"Unit {i}" for i in sorted(y.unique())],
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 ax=ax1)
        ax1.set_title('CART Decision Tree for TNPSC Question Classification', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # 2. Feature Importance
        ax2 = plt.subplot2grid((3, 3), (2, 0))
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax2.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        ax2.set_yticks(range(len(importance_df)))
        ax2.set_yticklabels(importance_df['feature'])
        ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance')
        
        # 3. Confusion Matrix
        ax3 = plt.subplot2grid((3, 3), (2, 1))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted Unit')
        ax3.set_ylabel('Actual Unit')
        
        # 4. Accuracy by Unit
        ax4 = plt.subplot2grid((3, 3), (2, 2))
        unit_accuracy = []
        unit_labels = []
        for unit in sorted(y_test.unique()):
            unit_mask = y_test == unit
            if unit_mask.sum() > 0:  # Only if there are samples for this unit
                unit_acc = accuracy_score(y_test[unit_mask], y_pred[unit_mask])
                unit_accuracy.append(unit_acc)
                unit_labels.append(f'Unit {unit}')
        
        bars = ax4.bar(range(len(unit_accuracy)), unit_accuracy, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(unit_accuracy))))
        ax4.set_title('Accuracy by Unit', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Unit')
        ax4.set_ylabel('Accuracy')
        ax4.set_xticks(range(len(unit_labels)))
        ax4.set_xticklabels(unit_labels, rotation=45)
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, unit_accuracy):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Decision Tree Performance:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Tree Depth: {dt_classifier.get_depth()}")
        print(f"   Number of Leaves: {dt_classifier.get_n_leaves()}")
        print("✅ RESULT: Decision tree visualizations created successfully!")
        
        return dt_classifier, accuracy, importance_df
    
    def experiment_9_ensemble_visual(self):
        """
        EXPERIMENT 9: Ensemble Learning with Beautiful Visualizations
        """
        print("\n" + "="*50)
        print("🎭 EXPERIMENT 9: ENSEMBLE LEARNING VISUALIZATION")
        print("="*50)
        
        df = self.questions_df.copy()
        
        # Use same features as decision tree
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        
        # Keyword features
        science_keywords = ['physics', 'chemistry', 'biology', 'science', 'atom', 'cell', 'energy']
        history_keywords = ['history', 'ancient', 'medieval', 'empire', 'dynasty', 'culture']
        geography_keywords = ['geography', 'river', 'mountain', 'climate', 'ocean', 'continent']
        
        df['science_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in science_keywords if keyword in x))
        df['history_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in history_keywords if keyword in x))
        df['geography_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in geography_keywords if keyword in x))
        
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers', 
                          'has_punctuation', 'avg_word_length', 'science_keywords', 
                          'history_keywords', 'geography_keywords']
        
        X = df[feature_columns]
        y = df['unit']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train ensemble models
        rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        ada_classifier = AdaBoostClassifier(n_estimators=30, learning_rate=1.0, random_state=42)
        dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)
        
        # Fit models
        rf_classifier.fit(X_train, y_train)
        ada_classifier.fit(X_train, y_train)
        dt_classifier.fit(X_train, y_train)
        
        # Predictions
        rf_pred = rf_classifier.predict(X_test)
        ada_pred = ada_classifier.predict(X_test)
        dt_pred = dt_classifier.predict(X_test)
        
        # Accuracies
        rf_accuracy = accuracy_score(y_test, rf_pred)
        ada_accuracy = accuracy_score(y_test, ada_pred)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        # Create ensemble visualization dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Ensemble Learning Comparison Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        models = ['Decision Tree', 'Random Forest', 'AdaBoost']
        accuracies = [dt_accuracy, rf_accuracy, ada_accuracy]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = axes[0,0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        axes[0,0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature Importance Comparison
        rf_importance = rf_classifier.feature_importances_
        ada_importance = ada_classifier.feature_importances_
        
        x = np.arange(len(feature_columns))
        width = 0.35
        
        bars1 = axes[0,1].bar(x - width/2, rf_importance, width, 
                             label='Random Forest', alpha=0.8, color='green')
        bars2 = axes[0,1].bar(x + width/2, ada_importance, width, 
                             label='AdaBoost', alpha=0.8, color='red')
        
        axes[0,1].set_xlabel('Features')
        axes[0,1].set_ylabel('Importance')
        axes[0,1].set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(feature_columns, rotation=45, ha='right')
        axes[0,1].legend()
        
        # 3. Confusion Matrix - Random Forest
        cm_rf = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0,2])
        axes[0,2].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Predicted Unit')
        axes[0,2].set_ylabel('Actual Unit')
        
        # 4. Confusion Matrix - AdaBoost
        cm_ada = confusion_matrix(y_test, ada_pred)
        sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Reds', ax=axes[1,0])
        axes[1,0].set_title('AdaBoost Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Predicted Unit')
        axes[1,0].set_ylabel('Actual Unit')
        
        # 5. Model Agreement Analysis
        agreement_rf_ada = (rf_pred == ada_pred).astype(int)
        agreement_counts = pd.Series(agreement_rf_ada).value_counts()
        
        labels = ['Disagree', 'Agree']
        sizes = [agreement_counts.get(0, 0), agreement_counts.get(1, 0)]
        colors_pie = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = axes[1,1].pie(sizes, labels=labels, autopct='%1.1f%%',
                                                colors=colors_pie, explode=(0.05, 0.05),
                                                shadow=True, startangle=90)
        axes[1,1].set_title('RF vs AdaBoost Agreement', fontsize=14, fontweight='bold')
        
        # 6. Performance Metrics Radar Chart
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate metrics for each model
        rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='weighted')
        ada_precision, ada_recall, ada_f1, _ = precision_recall_fscore_support(y_test, ada_pred, average='weighted')
        dt_precision, dt_recall, dt_f1, _ = precision_recall_fscore_support(y_test, dt_pred, average='weighted')
        
        # Create radar chart
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        rf_values = [rf_accuracy, rf_precision, rf_recall, rf_f1]
        ada_values = [ada_accuracy, ada_precision, ada_recall, ada_f1]
        dt_values = [dt_accuracy, dt_precision, dt_recall, dt_f1]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values to complete the circle
        rf_values += rf_values[:1]
        ada_values += ada_values[:1]
        dt_values += dt_values[:1]
        
        # Plot
        axes[1,2] = plt.subplot(2, 3, 6, projection='polar')
        axes[1,2].plot(angles, rf_values, 'o-', linewidth=2, label='Random Forest', color='green')
        axes[1,2].fill(angles, rf_values, alpha=0.25, color='green')
        axes[1,2].plot(angles, ada_values, 'o-', linewidth=2, label='AdaBoost', color='red')
        axes[1,2].fill(angles, ada_values, alpha=0.25, color='red')
        axes[1,2].plot(angles, dt_values, 'o-', linewidth=2, label='Decision Tree', color='blue')
        axes[1,2].fill(angles, dt_values, alpha=0.25, color='blue')
        
        # Add category labels
        axes[1,2].set_xticks(angles[:-1])
        axes[1,2].set_xticklabels(categories)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        axes[1,2].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Ensemble Learning Results:")
        print(f"   Decision Tree Accuracy: {dt_accuracy:.4f}")
        print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"   AdaBoost Accuracy: {ada_accuracy:.4f}")
        print(f"   Best Model: {'Random Forest' if rf_accuracy > ada_accuracy else 'AdaBoost'}")
        print("✅ RESULT: Ensemble learning visualizations created successfully!")
        
        return rf_classifier, ada_classifier, rf_accuracy, ada_accuracy
    
    def run_all_visual_experiments(self):
        """Run all experiments with beautiful visualizations"""
        print("="*60)
        print("🎨 RUNNING ALL VISUAL TNPSC ML EXPERIMENTS")
        print("="*60)
        
        results = {}
        
        try:
            # Run all experiments
            print("🚀 Starting visual experiments...")
            
            results['exp1'] = self.experiment_1_dataset_visualization()
            results['exp2'] = self.experiment_2_statistical_analysis()
            results['exp3'] = self.experiment_3_linear_regression_visual()
            results['exp4'] = self.experiment_4_classification_visual()
            results['exp5'] = self.experiment_5_clustering_visual()
            results['exp6'] = self.experiment_6_pca_visual()
            results['exp8'] = self.experiment_8_decision_tree_visual()
            results['exp9'] = self.experiment_9_ensemble_visual()
            
            print("\n" + "="*60)
            print("🎉 ALL VISUAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("📊 Beautiful graphs and visualizations generated!")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error in visual experiments: {e}")
            import traceback
            traceback.print_exc()
        
        return results

# Main execution
if __name__ == "__main__":
    # Initialize visual experiments
    visual_experiments = VisualTNPSCMLExperiments()
    
    print("\n🎨 Choose visualization option:")
    print("1-9: Run individual experiment with beautiful graphs")
    print("0: Run all experiments with stunning visualizations")
    print("q: Quit")
    
    choice = input("\nEnter your choice (0-9, q): ").strip().lower()
    
    if choice == "q":
        print("👋 Goodbye!")
    elif choice == "0":
        results = visual_experiments.run_all_visual_experiments()
    elif choice == "1":
        visual_experiments.experiment_1_dataset_visualization()
    elif choice == "2":
        visual_experiments.experiment_2_statistical_analysis()
    elif choice == "3":
        visual_experiments.experiment_3_linear_regression_visual()
    elif choice == "4":
        visual_experiments.experiment_4_classification_visual()
    elif choice == "5":
        visual_experiments.experiment_5_clustering_visual()
    elif choice == "6":
        visual_experiments.experiment_6_pca_visual()
    elif choice == "8":
        visual_experiments.experiment_8_decision_tree_visual()
    elif choice == "9":
        visual_experiments.experiment_9_ensemble_visual()
    else:
        print("❌ Invalid choice. Running all visual experiments...")
        results = visual_experiments.run_all_visual_experiments()