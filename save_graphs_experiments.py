#!/usr/bin/env python
# coding: utf-8

"""
AI-Powered TNPSC Exam Assistant - Visual Experiments with Saved Graphs
This version saves all graphs as PNG files so you can view them easily
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our TNPSC Assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS

# Set style for beautiful plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

print("🎨 TNPSC Visual ML Experiments - Saving Graphs as Images")
print("="*60)

class SaveGraphsExperiments:
    def __init__(self):
        self.assistant = TNPSCExamAssistant()
        self.questions_df = self.assistant.questions_df
        print("🚀 Experiments initialized - Graphs will be saved as PNG files!")
    
    def experiment_1_dataset_visualization(self):
        """Experiment 1: Dataset Visualization"""
        print("\n📊 EXPERIMENT 1: Creating Dataset Visualization...")
        
        df = self.questions_df.copy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TNPSC Dataset Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Questions per Unit
        unit_counts = df['unit'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unit_counts)))
        bars = axes[0,0].bar(unit_counts.index, unit_counts.values, color=colors)
        axes[0,0].set_title('Questions per Unit', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Unit Number')
        axes[0,0].set_ylabel('Number of Questions')
        
        # Add value labels
        for bar, value in zip(bars, unit_counts.values):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                          str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Difficulty Distribution
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
        
        # 3. Year-wise Distribution
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
        
        # 5. Question Length Distribution
        df['question_length'] = df['question'].str.len()
        axes[1,1].hist(df['question_length'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        axes[1,1].axvline(df['question_length'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["question_length"].mean():.1f}')
        axes[1,1].set_title('Question Length Distribution', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Question Length (characters)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        # 6. Unit Overview
        unit_names = [f"Unit {i}\n{UNITS[i]}" for i in unit_counts.index]
        y_pos = np.arange(len(unit_names))
        bars = axes[1,2].barh(y_pos, unit_counts.values, color=colors)
        axes[1,2].set_yticks(y_pos)
        axes[1,2].set_yticklabels(unit_names, fontsize=10)
        axes[1,2].set_title('Units Overview', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Number of Questions')
        
        plt.tight_layout()
        plt.savefig('experiment_1_dataset_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_1_dataset_visualization.png")
        plt.close()
    
    def experiment_3_linear_regression(self):
        """Experiment 3: Linear Regression Visualization"""
        print("\n📈 EXPERIMENT 3: Creating Linear Regression Visualization...")
        
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
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Actual vs Predicted
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
        
        # 3. Feature Importance
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=True)
        
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
        bars = axes[0,2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        axes[0,2].set_xlabel('Coefficient Value')
        axes[0,2].set_title('Feature Importance', fontsize=14, fontweight='bold')
        axes[0,2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Residual Distribution
        axes[1,0].hist(residuals, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Mean: {residuals.mean():.3f}')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1,0].legend()
        
        # 5. Feature vs Target (most important feature)
        most_important_idx = np.argmax(np.abs(model.coef_))
        most_important_feature = features[most_important_idx]
        axes[1,1].scatter(df[most_important_feature], df['difficulty_score'], 
                         alpha=0.7, color='orange', s=100, edgecolors='black')
        
        # Add trend line
        z = np.polyfit(df[most_important_feature], df['difficulty_score'], 1)
        p = np.poly1d(z)
        axes[1,1].plot(df[most_important_feature], p(df[most_important_feature]), "r--", alpha=0.8)
        axes[1,1].set_xlabel(most_important_feature)
        axes[1,1].set_ylabel('Difficulty Score')
        axes[1,1].set_title(f'{most_important_feature} vs Difficulty', fontsize=14, fontweight='bold')
        
        # 6. Model Performance Summary
        axes[1,2].axis('off')
        performance_text = f"""
        LINEAR REGRESSION RESULTS
        
        R² Score: {r2:.4f}
        Mean Squared Error: {mse:.4f}
        
        Feature Coefficients:
        """
        for feature, coef in zip(features, model.coef_):
            performance_text += f"\n{feature}: {coef:.4f}"
        
        axes[1,2].text(0.1, 0.9, performance_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_3_linear_regression.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_3_linear_regression.png")
        plt.close()
    
    def experiment_5_clustering(self):
        """Experiment 5: Clustering Visualization"""
        print("\n🎨 EXPERIMENT 5: Creating Clustering Visualization...")
        
        df = self.questions_df.copy()
        
        # Prepare features
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        df['question_length'] = df['question'].str.len()
        X_numerical = df[['question_length', 'year']].values
        
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        X = np.hstack([X_text, X_numerical_scaled])
        
        # Apply clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm_labels = gmm.fit_predict(X)
        gmm_proba = gmm.predict_proba(X)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        
        # 2. Gaussian Mixture Model
        scatter = axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, 
                                   cmap='plasma', alpha=0.7, s=100, edgecolors='black')
        axes[0,1].set_title('Gaussian Mixture Model', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('First Principal Component')
        axes[0,1].set_ylabel('Second Principal Component')
        
        # 3. Actual Unit Labels
        scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=df['unit'], 
                                   cmap='tab10', alpha=0.7, s=100, edgecolors='black')
        axes[1,0].set_title('Actual Unit Labels (Ground Truth)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('First Principal Component')
        axes[1,0].set_ylabel('Second Principal Component')
        
        # 4. GMM Confidence
        max_proba = np.max(gmm_proba, axis=1)
        scatter = axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c=max_proba, 
                                   cmap='coolwarm', alpha=0.7, s=100, edgecolors='black')
        axes[1,1].set_title('GMM Assignment Confidence', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('First Principal Component')
        axes[1,1].set_ylabel('Second Principal Component')
        
        plt.tight_layout()
        plt.savefig('experiment_5_clustering.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_5_clustering.png")
        plt.close()
    
    def experiment_8_decision_tree(self):
        """Experiment 8: Decision Tree Visualization"""
        print("\n🌳 EXPERIMENT 8: Creating Decision Tree Visualization...")
        
        df = self.questions_df.copy()
        
        # Create features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers', 'has_punctuation']
        X = df[feature_columns]
        y = df['unit']
        
        # Train decision tree
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt_classifier.fit(X_train, y_train)
        y_pred = dt_classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Decision Tree Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Decision Tree (Large plot)
        plot_tree(dt_classifier, 
                 feature_names=feature_columns,
                 class_names=[f"Unit {i}" for i in sorted(y.unique())],
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 ax=axes[0,0])
        axes[0,0].set_title('CART Decision Tree', fontsize=14, fontweight='bold')
        
        # 2. Feature Importance
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = axes[0,1].barh(range(len(importance_df)), importance_df['importance'], color=colors)
        axes[0,1].set_yticks(range(len(importance_df)))
        axes[0,1].set_yticklabels(importance_df['feature'])
        axes[0,1].set_title('Feature Importance', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Importance')
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Predicted Unit')
        axes[1,0].set_ylabel('Actual Unit')
        
        # 4. Performance Summary
        axes[1,1].axis('off')
        performance_text = f"""
        DECISION TREE RESULTS
        
        Overall Accuracy: {accuracy:.4f}
        Tree Depth: {dt_classifier.get_depth()}
        Number of Leaves: {dt_classifier.get_n_leaves()}
        
        Top Features by Importance:
        """
        for _, row in importance_df.tail(3).iterrows():
            performance_text += f"\n{row['feature']}: {row['importance']:.4f}"
        
        axes[1,1].text(0.1, 0.9, performance_text, transform=axes[1,1].transAxes, 
                      fontsize=14, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_8_decision_tree.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_8_decision_tree.png")
        plt.close()
    
    def experiment_9_ensemble(self):
        """Experiment 9: Ensemble Learning Visualization"""
        print("\n🎭 EXPERIMENT 9: Creating Ensemble Learning Visualization...")
        
        df = self.questions_df.copy()
        
        # Prepare features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers']
        X = df[feature_columns]
        y = df['unit']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train models
        rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        ada_classifier = AdaBoostClassifier(n_estimators=30, random_state=42)
        dt_classifier = DecisionTreeClassifier(random_state=42)
        
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
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        
        # 2. Random Forest Confusion Matrix
        cm_rf = confusion_matrix(y_test, rf_pred)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
        axes[0,1].set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Predicted Unit')
        axes[0,1].set_ylabel('Actual Unit')
        
        # 3. AdaBoost Confusion Matrix
        cm_ada = confusion_matrix(y_test, ada_pred)
        sns.heatmap(cm_ada, annot=True, fmt='d', cmap='Reds', ax=axes[1,0])
        axes[1,0].set_title('AdaBoost Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Predicted Unit')
        axes[1,0].set_ylabel('Actual Unit')
        
        # 4. Feature Importance Comparison
        rf_importance = rf_classifier.feature_importances_
        ada_importance = ada_classifier.feature_importances_
        
        x = np.arange(len(feature_columns))
        width = 0.35
        
        bars1 = axes[1,1].bar(x - width/2, rf_importance, width, 
                             label='Random Forest', alpha=0.8, color='green')
        bars2 = axes[1,1].bar(x + width/2, ada_importance, width, 
                             label='AdaBoost', alpha=0.8, color='red')
        
        axes[1,1].set_xlabel('Features')
        axes[1,1].set_ylabel('Importance')
        axes[1,1].set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(feature_columns, rotation=45, ha='right')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('experiment_9_ensemble.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_9_ensemble.png")
        plt.close()
    
    def experiment_2_statistical_analysis(self):
        """Experiment 2: Statistical Analysis Visualization"""
        print("\n📈 EXPERIMENT 2: Creating Statistical Analysis Visualization...")
        
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
        df['difficulty_num'] = df['difficulty'].map({'easy': 1, 'medium': 2, 'hard': 3})
        df['word_count'] = df['question'].str.split().str.len()
        
        corr_data = df[['unit', 'year', 'question_length', 'word_count', 'difficulty_num']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0,2])
        axes[0,2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 4. Stacked Bar Chart
        pivot_data = df.pivot_table(index='unit', columns='difficulty', values='year', aggfunc='count', fill_value=0)
        pivot_data.plot(kind='bar', stacked=True, ax=axes[1,0], color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1,0].set_title('Questions by Unit and Difficulty', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Unit')
        axes[1,0].set_ylabel('Number of Questions')
        axes[1,0].legend(title='Difficulty')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 5. Scatter Plot with Regression
        sns.scatterplot(data=df, x='question_length', y='word_count', hue='difficulty', 
                       size='unit', sizes=(50, 200), ax=axes[1,1])
        sns.regplot(data=df, x='question_length', y='word_count', scatter=False, 
                   color='red', ax=axes[1,1])
        axes[1,1].set_title('Question Length vs Word Count', fontsize=14, fontweight='bold')
        
        # 6. Statistical Summary
        axes[1,2].axis('off')
        stats_text = f"""
        STATISTICAL SUMMARY
        
        Total Questions: {len(df)}
        Units: {len(df['unit'].unique())}
        Years: {sorted(df['year'].unique())}
        
        Difficulty Distribution:
        """
        difficulty_counts = df['difficulty'].value_counts()
        for diff, count in difficulty_counts.items():
            percentage = (count / len(df)) * 100
            stats_text += f"\n{diff.capitalize()}: {count} ({percentage:.1f}%)"
        
        stats_text += f"""
        
        Question Length Stats:
        Mean: {df['question_length'].mean():.1f}
        Median: {df['question_length'].median():.1f}
        Std: {df['question_length'].std():.1f}
        """
        
        axes[1,2].text(0.1, 0.9, stats_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_2_statistical_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_2_statistical_analysis.png")
        plt.close()
    
    def experiment_4_classification(self):
        """Experiment 4: Classification Models Visualization"""
        print("\n🎯 EXPERIMENT 4: Creating Classification Visualization...")
        
        df = self.questions_df.copy()
        
        # Prepare features
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        df['question_length'] = df['question'].str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        X_numerical = df[['unit', 'year', 'question_length', 'has_numbers']].values
        
        X = np.hstack([X_text, X_numerical])
        y = df['difficulty']  # Multi-class classification
        
        # Split and train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        svm_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        svm_pred = svm_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        
        svm_accuracy = accuracy_score(y_test, svm_pred)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classification Models Comparison Dashboard', fontsize=20, fontweight='bold')
        
        # Get unique classes for consistent labeling
        unique_classes = sorted(list(set(y_test)))
        
        # 1. SVM Confusion Matrix
        cm_svm = confusion_matrix(y_test, svm_pred, labels=unique_classes)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_classes, 
                   yticklabels=unique_classes, ax=axes[0,0])
        axes[0,0].set_title('SVM Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. Logistic Regression Confusion Matrix
        cm_lr = confusion_matrix(y_test, lr_pred, labels=unique_classes)
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=unique_classes, 
                   yticklabels=unique_classes, ax=axes[0,1])
        axes[0,1].set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. Model Accuracy Comparison
        models = ['SVM', 'Logistic Regression']
        accuracies = [svm_accuracy, lr_accuracy]
        colors = ['#3498db', '#2ecc71']
        
        bars = axes[0,2].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        axes[0,2].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0,2].set_ylabel('Accuracy')
        axes[0,2].set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Prediction Distribution
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'SVM_Pred': svm_pred,
            'LR_Pred': lr_pred
        })
        
        # Ensure all counts have the same classes
        actual_counts = pred_df['Actual'].value_counts().reindex(unique_classes, fill_value=0)
        svm_counts = pred_df['SVM_Pred'].value_counts().reindex(unique_classes, fill_value=0)
        lr_counts = pred_df['LR_Pred'].value_counts().reindex(unique_classes, fill_value=0)
        
        x = np.arange(len(unique_classes))
        width = 0.25
        
        axes[1,0].bar(x - width, actual_counts.values, width, label='Actual', alpha=0.8)
        axes[1,0].bar(x, svm_counts.values, width, label='SVM', alpha=0.8)
        axes[1,0].bar(x + width, lr_counts.values, width, label='LR', alpha=0.8)
        
        axes[1,0].set_xlabel('Difficulty Level')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Prediction Distribution Comparison', fontsize=14, fontweight='bold')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(unique_classes)
        axes[1,0].legend()
        
        # 5. Classification Report Visualization
        from sklearn.metrics import precision_recall_fscore_support
        
        # Get unique classes from test set
        unique_classes = sorted(list(set(y_test)))
        
        svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(y_test, svm_pred, average=None, labels=unique_classes)
        lr_precision, lr_recall, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average=None, labels=unique_classes)
        
        x = np.arange(len(unique_classes))
        width = 0.2
        
        axes[1,1].bar(x - width*1.5, svm_precision, width, label='SVM Precision', alpha=0.8)
        axes[1,1].bar(x - width/2, svm_recall, width, label='SVM Recall', alpha=0.8)
        axes[1,1].bar(x + width/2, lr_precision, width, label='LR Precision', alpha=0.8)
        axes[1,1].bar(x + width*1.5, lr_recall, width, label='LR Recall', alpha=0.8)
        
        axes[1,1].set_xlabel('Classes')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Precision & Recall Comparison', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(unique_classes)
        axes[1,1].legend()
        axes[1,1].set_ylim(0, 1)
        
        # 6. Performance Summary
        axes[1,2].axis('off')
        performance_text = f"""
        CLASSIFICATION RESULTS
        
        SVM Performance:
        Accuracy: {svm_accuracy:.4f}
        
        Logistic Regression Performance:
        Accuracy: {lr_accuracy:.4f}
        
        Best Model: {'SVM' if svm_accuracy > lr_accuracy else 'Logistic Regression'}
        
        Dataset Info:
        Total Samples: {len(y_test)}
        Features: {X.shape[1]}
        Classes: {len(set(y_test))}
        """
        
        axes[1,2].text(0.1, 0.9, performance_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_4_classification.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_4_classification.png")
        plt.close()
    
    def experiment_6_pca(self):
        """Experiment 6: PCA Analysis Visualization"""
        print("\n🔍 EXPERIMENT 6: Creating PCA Visualization...")
        
        df = self.questions_df.copy()
        
        # Prepare features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        
        X_numerical = df[['unit', 'year', 'question_length', 'word_count', 
                         'has_numbers', 'has_punctuation', 'avg_word_length']].values
        
        X = np.hstack([X_text, X_numerical])
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Create visualization
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
        
        # 6. PCA Summary
        axes[1,2].axis('off')
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        pca_text = f"""
        PCA ANALYSIS SUMMARY
        
        Original Dimensions: {X.shape[1]}
        
        Variance Explained:
        PC1: {explained_variance_ratio[0]:.1%}
        PC2: {explained_variance_ratio[1]:.1%}
        PC3: {explained_variance_ratio[2]:.1%}
        
        Components for 95% variance: {n_components_95}
        Components for 90% variance: {np.argmax(cumulative_variance >= 0.90) + 1}
        
        Total samples: {X.shape[0]}
        """
        
        axes[1,2].text(0.1, 0.9, pca_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_6_pca.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_6_pca.png")
        plt.close()
    
    def experiment_7_hmm(self):
        """Experiment 7: HMM Sequential Analysis Visualization"""
        print("\n🔄 EXPERIMENT 7: Creating HMM Sequential Analysis Visualization...")
        
        df = self.questions_df.copy().sort_values(['unit', 'year'])
        
        # Create difficulty sequences for each unit
        difficulty_sequences = {}
        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit].sort_values('year')
            difficulty_sequences[unit] = unit_data['difficulty'].tolist()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Hidden Markov Model Sequential Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Difficulty Sequences by Unit
        colors = plt.cm.Set3(np.linspace(0, 1, len(difficulty_sequences)))
        for i, (unit, sequence) in enumerate(difficulty_sequences.items()):
            difficulty_numeric = [{'easy': 1, 'medium': 2, 'hard': 3}[d] for d in sequence]
            axes[0,0].plot(range(len(difficulty_numeric)), difficulty_numeric, 
                          'o-', label=f'Unit {unit}', color=colors[i], linewidth=2, markersize=8)
        
        axes[0,0].set_title('Difficulty Sequences by Unit', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Question Index')
        axes[0,0].set_ylabel('Difficulty Level')
        axes[0,0].set_yticks([1, 2, 3])
        axes[0,0].set_yticklabels(['Easy', 'Medium', 'Hard'])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Transition Matrix Heatmap
        # Calculate transition probabilities
        all_sequences = []
        for seq in difficulty_sequences.values():
            all_sequences.extend(seq)
        
        # Create transition matrix
        difficulties = ['easy', 'medium', 'hard']
        transition_matrix = np.zeros((3, 3))
        
        for i in range(len(all_sequences) - 1):
            current = difficulties.index(all_sequences[i])
            next_diff = difficulties.index(all_sequences[i + 1])
            transition_matrix[current, next_diff] += 1
        
        # Normalize to probabilities
        for i in range(3):
            if transition_matrix[i].sum() > 0:
                transition_matrix[i] = transition_matrix[i] / transition_matrix[i].sum()
        
        sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=difficulties, yticklabels=difficulties, ax=axes[0,1])
        axes[0,1].set_title('Difficulty Transition Matrix', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Next Difficulty')
        axes[0,1].set_ylabel('Current Difficulty')
        
        # 3. Sequence Length Distribution
        seq_lengths = [len(seq) for seq in difficulty_sequences.values()]
        axes[0,2].bar(range(len(seq_lengths)), seq_lengths, 
                     color=colors[:len(seq_lengths)], alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Sequence Lengths by Unit', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Unit')
        axes[0,2].set_ylabel('Number of Questions')
        axes[0,2].set_xticks(range(len(seq_lengths)))
        axes[0,2].set_xticklabels([f'Unit {unit}' for unit in difficulty_sequences.keys()])
        
        # 4. Difficulty Distribution Over Time
        df_sorted = df.sort_values('year')
        year_difficulty = df_sorted.groupby(['year', 'difficulty']).size().unstack(fill_value=0)
        year_difficulty.plot(kind='bar', stacked=True, ax=axes[1,0], 
                           color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1,0].set_title('Difficulty Distribution Over Years', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Number of Questions')
        axes[1,0].legend(title='Difficulty')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 5. Unit-wise Difficulty Patterns
        unit_difficulty = df.groupby(['unit', 'difficulty']).size().unstack(fill_value=0)
        unit_difficulty_pct = unit_difficulty.div(unit_difficulty.sum(axis=1), axis=0)
        
        sns.heatmap(unit_difficulty_pct, annot=True, fmt='.2f', cmap='RdYlBu',
                   ax=axes[1,1])
        axes[1,1].set_title('Unit-wise Difficulty Patterns (%)', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Difficulty')
        axes[1,1].set_ylabel('Unit')
        
        # 6. HMM Analysis Summary
        axes[1,2].axis('off')
        
        # Calculate some statistics
        total_transitions = sum(len(seq) - 1 for seq in difficulty_sequences.values() if len(seq) > 1)
        most_common_transition = None
        max_prob = 0
        
        for i, from_diff in enumerate(difficulties):
            for j, to_diff in enumerate(difficulties):
                if transition_matrix[i, j] > max_prob:
                    max_prob = transition_matrix[i, j]
                    most_common_transition = f"{from_diff} → {to_diff}"
        
        hmm_text = f"""
        HMM SEQUENTIAL ANALYSIS
        
        Total Units Analyzed: {len(difficulty_sequences)}
        Total Questions: {len(df)}
        Total Transitions: {total_transitions}
        
        Most Common Transition:
        {most_common_transition}
        (Probability: {max_prob:.3f})
        
        Difficulty Distribution:
        """
        
        difficulty_counts = df['difficulty'].value_counts()
        for diff, count in difficulty_counts.items():
            percentage = (count / len(df)) * 100
            hmm_text += f"\n{diff.capitalize()}: {percentage:.1f}%"
        
        axes[1,2].text(0.1, 0.9, hmm_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('experiment_7_hmm.png', dpi=300, bbox_inches='tight')
        print("✅ Graph saved: experiment_7_hmm.png")
        plt.close()
    
    def run_all_experiments(self):
        """Run all experiments and save graphs"""
        print("🚀 Running all visual experiments and saving graphs...")
        
        self.experiment_1_dataset_visualization()
        self.experiment_2_statistical_analysis()
        self.experiment_3_linear_regression()
        self.experiment_4_classification()
        self.experiment_5_clustering()
        self.experiment_6_pca()
        self.experiment_7_hmm()
        self.experiment_8_decision_tree()
        self.experiment_9_ensemble()
        
        print("\n🎉 ALL 9 EXPERIMENTS COMPLETED!")
        print("📁 All graphs saved as PNG files in the current directory:")
        print("   - experiment_1_dataset_visualization.png")
        print("   - experiment_2_statistical_analysis.png")
        print("   - experiment_3_linear_regression.png")
        print("   - experiment_4_classification.png")
        print("   - experiment_5_clustering.png")
        print("   - experiment_6_pca.png")
        print("   - experiment_7_hmm.png")
        print("   - experiment_8_decision_tree.png")
        print("   - experiment_9_ensemble.png")

# Main execution
if __name__ == "__main__":
    experiments = SaveGraphsExperiments()
    
    print("\n🎨 Choose option:")
    print("0: Run ALL 9 experiments and save graphs")
    print("1: Dataset visualization")
    print("2: Statistical analysis")
    print("3: Linear regression")
    print("4: Classification models")
    print("5: Clustering")
    print("6: PCA analysis")
    print("7: HMM sequential analysis")
    print("8: Decision tree")
    print("9: Ensemble learning")
    
    choice = input("\nEnter choice: ").strip()
    
    if choice == "0":
        experiments.run_all_experiments()
    elif choice == "1":
        experiments.experiment_1_dataset_visualization()
    elif choice == "2":
        experiments.experiment_2_statistical_analysis()
    elif choice == "3":
        experiments.experiment_3_linear_regression()
    elif choice == "4":
        experiments.experiment_4_classification()
    elif choice == "5":
        experiments.experiment_5_clustering()
    elif choice == "6":
        experiments.experiment_6_pca()
    elif choice == "7":
        experiments.experiment_7_hmm()
    elif choice == "8":
        experiments.experiment_8_decision_tree()
    elif choice == "9":
        experiments.experiment_9_ensemble()
    else:
        print("Running all experiments...")
        experiments.run_all_experiments()