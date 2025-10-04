#!/usr/bin/env python
# coding: utf-8

"""
AI-Powered Personalized Exam Preparation Assistant for Competitive Exams
Machine Learning Laboratory Experiments (1-9)
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import our existing TNPSC Assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS, load_questions, load_syllabus

print("="*60)
print("AI-POWERED TNPSC EXAM PREPARATION ASSISTANT")
print("MACHINE LEARNING LABORATORY EXPERIMENTS")
print("="*60)

class TNPSCMLExperiments:
    """Class containing all 9 ML experiments for TNPSC Exam Assistant"""
    
    def __init__(self):
        self.assistant = TNPSCExamAssistant()
        self.questions_df = self.assistant.questions_df
        self.syllabus = self.assistant.syllabus
        print("TNPSC ML Experiments initialized successfully!")
    
    def experiment_1_load_and_view_dataset(self):
        """
        EXPERIMENT 1: Load and View the TNPSC Dataset
        AIM: To implement a program to load and view the TNPSC questions dataset
        """
        print("\n" + "="*50)
        print("EXPERIMENT 1: LOAD AND VIEW TNPSC DATASET")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Load TNPSC questions dataset")
        print("2. Create a duplicate dataset for safety")
        print("3. Display basic information about the dataset")
        print("4. Show sample questions and statistics")
        
        print("\nPROGRAM EXECUTION:")
        
        # Load the dataset
        print(f"Original dataset shape: {self.questions_df.shape}")
        
        # Create duplicate
        tnpsc_data = self.questions_df.copy()
        
        # Display basic info
        print(f"\nDataset columns: {list(tnpsc_data.columns)}")
        print(f"Total questions: {len(tnpsc_data)}")
        print(f"Units covered: {sorted(tnpsc_data['unit'].unique())}")
        
        # Display first 10 questions
        print("\nFirst 10 questions:")
        display_cols = ['unit', 'question', 'difficulty', 'year']
        print(tnpsc_data[display_cols].head(10))
        
        # Unit distribution
        print("\nQuestions per unit:")
        unit_counts = tnpsc_data['unit'].value_counts().sort_index()
        for unit, count in unit_counts.items():
            print(f"Unit {unit} ({UNITS[unit]}): {count} questions")
        
        print("\nRESULT: TNPSC dataset loaded and analyzed successfully!")
        return tnpsc_data
    
    def experiment_2_dataset_statistics(self):
        """
        EXPERIMENT 2: Display Summary and Statistics of TNPSC Dataset
        AIM: To display the summary and statistics of the TNPSC questions dataset
        """
        print("\n" + "="*50)
        print("EXPERIMENT 2: DATASET STATISTICS")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Load TNPSC dataset")
        print("2. Calculate descriptive statistics")
        print("3. Display difficulty distribution")
        print("4. Show year-wise question distribution")
        
        print("\nPROGRAM EXECUTION:")
        
        # Basic statistics
        print("Dataset Overview:")
        print(f"Total Questions: {len(self.questions_df)}")
        print(f"Units: {len(self.questions_df['unit'].unique())}")
        print(f"Years covered: {sorted(self.questions_df['year'].unique())}")
        
        # Difficulty distribution
        print("\nDifficulty Distribution:")
        difficulty_counts = self.questions_df['difficulty'].value_counts()
        for diff, count in difficulty_counts.items():
            percentage = (count / len(self.questions_df)) * 100
            print(f"{diff.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Year distribution
        print("\nYear-wise Distribution:")
        year_counts = self.questions_df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"Year {year}: {count} questions")
        
        # Unit-wise statistics
        print("\nUnit-wise Statistics:")
        unit_stats = self.questions_df.groupby('unit').agg({
            'question': 'count',
            'difficulty': lambda x: x.mode().iloc[0],
            'year': 'mean'
        }).round(1)
        unit_stats.columns = ['Question_Count', 'Most_Common_Difficulty', 'Avg_Year']
        
        for unit in sorted(unit_stats.index):
            stats = unit_stats.loc[unit]
            print(f"Unit {unit} ({UNITS[unit]}): {stats['Question_Count']} questions, "
                  f"Difficulty: {stats['Most_Common_Difficulty']}, Avg Year: {stats['Avg_Year']}")
        
        print("\nRESULT: Dataset statistics calculated and displayed successfully!")
        return unit_stats
    
    def experiment_3_linear_regression_prediction(self):
        """
        EXPERIMENT 3: Linear Regression for Question Difficulty Prediction
        AIM: To implement linear regression to predict question difficulty scores
        """
        print("\n" + "="*50)
        print("EXPERIMENT 3: LINEAR REGRESSION PREDICTION")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare features from question text and metadata")
        print("2. Convert difficulty to numerical scores")
        print("3. Split data into training and testing sets")
        print("4. Train linear regression model")
        print("5. Make predictions and evaluate performance")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features
        df = self.questions_df.copy()
        
        # Convert difficulty to numerical scores
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
        df['difficulty_score'] = df['difficulty'].map(difficulty_map)
        
        # Create features
        df['question_length'] = df['question'].str.len()
        df['option_a_length'] = df['option_a'].str.len()
        df['option_b_length'] = df['option_b'].str.len()
        df['option_c_length'] = df['option_c'].str.len()
        df['option_d_length'] = df['option_d'].str.len()
        df['avg_option_length'] = (df['option_a_length'] + df['option_b_length'] + 
                                  df['option_c_length'] + df['option_d_length']) / 4
        
        # Features and target
        features = ['unit', 'year', 'question_length', 'avg_option_length']
        X = df[features]
        y = df['difficulty_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance
        print(f"\nFeature Coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        
        # Sample predictions
        print(f"\nSample Predictions vs Actual:")
        for i in range(min(5, len(y_test))):
            print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]}")
        
        print("\nRESULT: Linear regression model trained and evaluated successfully!")
        return model, mse, r2
    
    def experiment_4_1_bayesian_logistic_regression(self):
        """
        EXPERIMENT 4.1: Bayesian Logistic Regression for Unit Classification
        AIM: To implement Bayesian logistic regression for classifying questions by unit
        """
        print("\n" + "="*50)
        print("EXPERIMENT 4.1: BAYESIAN LOGISTIC REGRESSION")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare text features using TF-IDF")
        print("2. Create binary classification problem (Science vs Non-Science)")
        print("3. Implement Bayesian inference using sklearn approximation")
        print("4. Train and evaluate the model")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare data
        df = self.questions_df.copy()
        
        # Create binary classification: Science (Unit 1) vs Non-Science
        df['is_science'] = (df['unit'] == 1).astype(int)
        
        # Create text features using TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Add numerical features
        df['question_length'] = df['question'].str.len()
        X_numerical = df[['question_length', 'year']].values
        
        # Combine features
        X = np.hstack([X_text, X_numerical])
        y = df['is_science']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Bayesian-inspired Logistic Regression (using regularization as prior)
        model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Science', 'Science']))
        
        # Show prediction probabilities for first 5 samples
        print(f"\nSample Prediction Probabilities:")
        for i in range(min(5, len(y_test))):
            prob_non_science, prob_science = y_pred_proba[i]
            actual = "Science" if y_test.iloc[i] == 1 else "Non-Science"
            predicted = "Science" if y_pred[i] == 1 else "Non-Science"
            print(f"Sample {i+1}: Predicted={predicted} (Science: {prob_science:.3f}), Actual={actual}")
        
        print("\nRESULT: Bayesian logistic regression implemented and evaluated successfully!")
        return model, accuracy
    
    def experiment_4_2_svm_classification(self):
        """
        EXPERIMENT 4.2: SVM for Question Difficulty Classification
        AIM: To implement SVM for classifying question difficulty levels
        """
        print("\n" + "="*50)
        print("EXPERIMENT 4.2: SVM CLASSIFICATION")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare features from question text and metadata")
        print("2. Create multi-class classification for difficulty levels")
        print("3. Train SVM model with RBF kernel")
        print("4. Evaluate model performance")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare data
        df = self.questions_df.copy()
        
        # Create features
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Additional features
        df['question_length'] = df['question'].str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        X_numerical = df[['unit', 'year', 'question_length', 'has_numbers']].values
        
        # Combine features
        X = np.hstack([X_text, X_numerical])
        y = df['difficulty']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train SVM
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm_model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Show some predictions
        print(f"\nSample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"Question: {df.iloc[X_test.index[i]]['question'][:50]}...")
            print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
            print("-" * 50)
        
        print("\nRESULT: SVM classification model trained and evaluated successfully!")
        return svm_model, accuracy
    
    def experiment_5_1_kmeans_clustering(self):
        """
        EXPERIMENT 5.1: K-Means Clustering for Question Categorization
        AIM: To implement K-means clustering to categorize TNPSC questions
        """
        print("\n" + "="*50)
        print("EXPERIMENT 5.1: K-MEANS CLUSTERING")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Create feature vectors from question text")
        print("2. Apply K-means clustering with k=8 (number of units)")
        print("3. Analyze clusters and their characteristics")
        print("4. Visualize clustering results")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features
        df = self.questions_df.copy()
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Add numerical features
        df['question_length'] = df['question'].str.len()
        X_numerical = df[['question_length', 'year']].values
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        
        # Combine features
        X = np.hstack([X_text, X_numerical_scaled])
        
        # Apply K-means
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster labels to dataframe
        df['cluster'] = clusters
        
        # Analyze clusters
        print("Cluster Analysis:")
        for cluster_id in range(8):
            cluster_data = df[df['cluster'] == cluster_id]
            most_common_unit = cluster_data['unit'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            most_common_difficulty = cluster_data['difficulty'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            
            print(f"Cluster {cluster_id}: {len(cluster_data)} questions")
            print(f"  Most common unit: {most_common_unit} ({UNITS.get(most_common_unit, 'Unknown')})")
            print(f"  Most common difficulty: {most_common_difficulty}")
            print(f"  Units distribution: {dict(cluster_data['unit'].value_counts())}")
            print()
        
        # Calculate cluster purity (how well clusters match actual units)
        cluster_purity = []
        for cluster_id in range(8):
            cluster_data = df[df['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                most_common_count = cluster_data['unit'].value_counts().iloc[0]
                purity = most_common_count / len(cluster_data)
                cluster_purity.append(purity)
        
        avg_purity = np.mean(cluster_purity)
        print(f"Average Cluster Purity: {avg_purity:.4f}")
        
        # Visualize using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 5))
        
        # Plot clusters
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title('K-Means Clustering Results')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # Plot actual units
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['unit'], cmap='tab10', alpha=0.6)
        plt.title('Actual Unit Labels')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        print("\nRESULT: K-means clustering applied successfully to TNPSC questions!")
        return kmeans, clusters, avg_purity  
  
    def experiment_5_2_gaussian_mixture_models(self):
        """
        EXPERIMENT 5.2: Gaussian Mixture Models for Question Categorization
        AIM: To implement GMM to categorize TNPSC questions with probabilistic clustering
        """
        print("\n" + "="*50)
        print("EXPERIMENT 5.2: GAUSSIAN MIXTURE MODELS")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare feature vectors from question text and metadata")
        print("2. Apply Gaussian Mixture Model with 8 components")
        print("3. Analyze probabilistic cluster assignments")
        print("4. Compare with actual unit labels")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features (same as K-means)
        df = self.questions_df.copy()
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Add numerical features
        df['question_length'] = df['question'].str.len()
        X_numerical = df[['question_length', 'year']].values
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        
        # Combine features
        X = np.hstack([X_text, X_numerical_scaled])
        
        # Apply Gaussian Mixture Model
        gmm = GaussianMixture(n_components=8, random_state=42, covariance_type='full')
        gmm.fit(X)
        
        # Get cluster assignments and probabilities
        clusters = gmm.predict(X)
        probabilities = gmm.predict_proba(X)
        
        # Add to dataframe
        df['gmm_cluster'] = clusters
        df['max_probability'] = np.max(probabilities, axis=1)
        
        # Analyze clusters
        print("GMM Cluster Analysis:")
        for cluster_id in range(8):
            cluster_data = df[df['gmm_cluster'] == cluster_id]
            if len(cluster_data) > 0:
                most_common_unit = cluster_data['unit'].mode().iloc[0]
                avg_probability = cluster_data['max_probability'].mean()
                
                print(f"Cluster {cluster_id}: {len(cluster_data)} questions")
                print(f"  Most common unit: {most_common_unit} ({UNITS[most_common_unit]})")
                print(f"  Average assignment probability: {avg_probability:.4f}")
                print(f"  Units distribution: {dict(cluster_data['unit'].value_counts())}")
                print()
        
        # Calculate AIC and BIC
        aic = gmm.aic(X)
        bic = gmm.bic(X)
        print(f"Model Selection Metrics:")
        print(f"AIC: {aic:.2f}")
        print(f"BIC: {bic:.2f}")
        
        # Visualize using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 5))
        
        # Plot GMM clusters
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.title('GMM Clustering Results')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # Plot probability confidence
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['max_probability'], 
                            cmap='plasma', alpha=0.6)
        plt.title('Assignment Probability Confidence')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Max Probability')
        
        plt.tight_layout()
        plt.show()
        
        print("\nRESULT: Gaussian Mixture Model clustering applied successfully!")
        return gmm, clusters, aic, bic
    
    def experiment_5_3_hierarchical_clustering(self):
        """
        EXPERIMENT 5.3: Hierarchical Clustering for Question Categorization
        AIM: To implement hierarchical clustering to categorize TNPSC questions
        """
        print("\n" + "="*50)
        print("EXPERIMENT 5.3: HIERARCHICAL CLUSTERING")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare feature vectors from question text")
        print("2. Calculate distance matrix between questions")
        print("3. Apply hierarchical clustering using linkage methods")
        print("4. Create dendrogram and extract clusters")
        
        print("\nPROGRAM EXECUTION:")
        
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        from scipy.spatial.distance import pdist
        
        # Prepare features (using smaller subset for computational efficiency)
        df = self.questions_df.copy()
        
        # Use a subset for demonstration (hierarchical clustering is computationally expensive)
        df_sample = df.sample(n=min(50, len(df)), random_state=42)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        X_text = vectorizer.fit_transform(df_sample['question']).toarray()
        
        # Add numerical features
        df_sample['question_length'] = df_sample['question'].str.len()
        X_numerical = df_sample[['question_length', 'year']].values
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)
        
        # Combine features
        X = np.hstack([X_text, X_numerical_scaled])
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(X, method='ward')
        
        # Extract clusters
        clusters = fcluster(linkage_matrix, t=8, criterion='maxclust')
        df_sample['hierarchical_cluster'] = clusters
        
        # Analyze clusters
        print("Hierarchical Clustering Analysis:")
        for cluster_id in range(1, 9):  # fcluster returns 1-indexed clusters
            cluster_data = df_sample[df_sample['hierarchical_cluster'] == cluster_id]
            if len(cluster_data) > 0:
                most_common_unit = cluster_data['unit'].mode().iloc[0]
                
                print(f"Cluster {cluster_id}: {len(cluster_data)} questions")
                print(f"  Most common unit: {most_common_unit} ({UNITS[most_common_unit]})")
                print(f"  Units in cluster: {list(cluster_data['unit'].unique())}")
                print()
        
        # Create dendrogram
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        # Visualize clusters using PCA
        plt.subplot(2, 1, 2)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.title('Hierarchical Clustering Results (PCA Visualization)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        print("\nRESULT: Hierarchical clustering applied successfully to TNPSC questions!")
        return linkage_matrix, clusters
    
    def experiment_6_pca_analysis(self):
        """
        EXPERIMENT 6: Principal Component Analysis on TNPSC Questions
        AIM: To perform PCA on TNPSC question features for dimensionality reduction
        """
        print("\n" + "="*50)
        print("EXPERIMENT 6: PRINCIPAL COMPONENT ANALYSIS")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare high-dimensional feature vectors from questions")
        print("2. Standardize the features")
        print("3. Apply PCA to reduce dimensionality")
        print("4. Analyze explained variance and principal components")
        print("5. Visualize results in reduced dimensions")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features
        df = self.questions_df.copy()
        
        # Create comprehensive feature set
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        X_text = vectorizer.fit_transform(df['question']).toarray()
        
        # Additional features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        
        X_numerical = df[['unit', 'year', 'question_length', 'word_count', 
                         'has_numbers', 'has_punctuation', 'avg_word_length']].values
        
        # Combine all features
        X = np.hstack([X_text, X_numerical])
        
        print(f"Original feature dimensions: {X.shape}")
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"Explained variance by first 10 components:")
        for i in range(min(10, len(explained_variance_ratio))):
            print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} ({cumulative_variance[i]:.4f} cumulative)")
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"\nComponents needed for 95% variance: {n_components_95}")
        
        # Reduce to optimal dimensions
        pca_reduced = PCA(n_components=n_components_95)
        X_reduced = pca_reduced.fit_transform(X_scaled)
        
        print(f"Reduced dimensions: {X_reduced.shape}")
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Scree plot
        plt.subplot(2, 3, 1)
        plt.plot(range(1, min(21, len(explained_variance_ratio) + 1)), 
                explained_variance_ratio[:20], 'bo-')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        
        # Cumulative variance plot
        plt.subplot(2, 3, 2)
        plt.plot(range(1, min(21, len(cumulative_variance) + 1)), 
                cumulative_variance[:20], 'ro-')
        plt.axhline(y=0.95, color='k', linestyle='--', label='95% Variance')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Ratio')
        plt.legend()
        plt.grid(True)
        
        # 2D visualization by unit
        plt.subplot(2, 3, 3)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['unit'], cmap='tab10', alpha=0.6)
        plt.title('PCA: Questions by Unit')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # 2D visualization by difficulty
        plt.subplot(2, 3, 4)
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        difficulty_numeric = df['difficulty'].map(difficulty_map)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=difficulty_numeric, cmap='viridis', alpha=0.6)
        plt.title('PCA: Questions by Difficulty')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, ticks=[0, 1, 2], label='Difficulty')
        
        # 2D visualization by year
        plt.subplot(2, 3, 5)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['year'], cmap='plasma', alpha=0.6)
        plt.title('PCA: Questions by Year')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        
        # Feature importance in first PC
        plt.subplot(2, 3, 6)
        feature_names = (list(vectorizer.get_feature_names_out()) + 
                        ['unit', 'year', 'question_length', 'word_count', 
                         'has_numbers', 'has_punctuation', 'avg_word_length'])
        
        # Show top 10 features contributing to PC1
        pc1_importance = np.abs(pca.components_[0])
        top_features_idx = np.argsort(pc1_importance)[-10:]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importance = pc1_importance[top_features_idx]
        
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.title('Top Features in PC1')
        plt.xlabel('Absolute Loading')
        
        plt.tight_layout()
        plt.show()
        
        print("\nRESULT: PCA analysis completed successfully!")
        return pca, X_reduced, explained_variance_ratio
    
    def experiment_7_hmm_sequential_prediction(self):
        """
        EXPERIMENT 7: Hidden Markov Model for Sequential Question Difficulty Prediction
        AIM: To implement HMM to predict sequential patterns in question difficulty
        """
        print("\n" + "="*50)
        print("EXPERIMENT 7: HIDDEN MARKOV MODEL")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Create sequences of question difficulties by unit and year")
        print("2. Define states (difficulty levels) and observations")
        print("3. Estimate HMM parameters from data")
        print("4. Use Viterbi algorithm to predict most likely sequences")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare sequential data
        df = self.questions_df.copy().sort_values(['unit', 'year'])
        
        # Create difficulty sequences for each unit
        difficulty_sequences = {}
        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit].sort_values('year')
            difficulty_sequences[unit] = unit_data['difficulty'].tolist()
        
        # Simple HMM implementation
        class SimpleHMM:
            def __init__(self, states, observations):
                self.states = states
                self.observations = observations
                self.n_states = len(states)
                self.n_obs = len(observations)
                
                # Initialize parameters
                self.start_prob = np.ones(self.n_states) / self.n_states
                self.trans_prob = np.ones((self.n_states, self.n_states)) / self.n_states
                self.emit_prob = np.ones((self.n_states, self.n_obs)) / self.n_obs
            
            def fit(self, sequences):
                """Estimate parameters from sequences using simple counting"""
                # Count transitions and emissions
                trans_counts = np.zeros((self.n_states, self.n_states))
                emit_counts = np.zeros((self.n_states, self.n_obs))
                start_counts = np.zeros(self.n_states)
                
                state_map = {state: i for i, state in enumerate(self.states)}
                obs_map = {obs: i for i, obs in enumerate(self.observations)}
                
                for seq in sequences:
                    if len(seq) == 0:
                        continue
                    
                    # Convert to indices
                    seq_indices = [obs_map[obs] for obs in seq if obs in obs_map]
                    
                    if len(seq_indices) == 0:
                        continue
                    
                    # Assume states = observations for simplicity
                    state_seq = seq_indices
                    
                    # Count start state
                    start_counts[state_seq[0]] += 1
                    
                    # Count transitions and emissions
                    for t in range(len(state_seq)):
                        emit_counts[state_seq[t], seq_indices[t]] += 1
                        if t > 0:
                            trans_counts[state_seq[t-1], state_seq[t]] += 1
                
                # Normalize to get probabilities
                self.start_prob = start_counts / (start_counts.sum() + 1e-10)
                
                for i in range(self.n_states):
                    trans_sum = trans_counts[i].sum()
                    if trans_sum > 0:
                        self.trans_prob[i] = trans_counts[i] / trans_sum
                    
                    emit_sum = emit_counts[i].sum()
                    if emit_sum > 0:
                        self.emit_prob[i] = emit_counts[i] / emit_sum
            
            def viterbi(self, obs_sequence):
                """Viterbi algorithm for finding most likely state sequence"""
                if len(obs_sequence) == 0:
                    return [], 0
                
                obs_map = {obs: i for i, obs in enumerate(self.observations)}
                obs_indices = [obs_map.get(obs, 0) for obs in obs_sequence]
                
                T = len(obs_indices)
                
                # Initialize
                viterbi_table = np.zeros((self.n_states, T))
                backpointer = np.zeros((self.n_states, T), dtype=int)
                
                # Initialization
                for s in range(self.n_states):
                    viterbi_table[s, 0] = self.start_prob[s] * self.emit_prob[s, obs_indices[0]]
                
                # Recursion
                for t in range(1, T):
                    for s in range(self.n_states):
                        probs = viterbi_table[:, t-1] * self.trans_prob[:, s] * self.emit_prob[s, obs_indices[t]]
                        viterbi_table[s, t] = np.max(probs)
                        backpointer[s, t] = np.argmax(probs)
                
                # Termination
                best_path_prob = np.max(viterbi_table[:, T-1])
                best_last_state = np.argmax(viterbi_table[:, T-1])
                
                # Backtracking
                best_path = np.zeros(T, dtype=int)
                best_path[-1] = best_last_state
                for t in range(T-2, -1, -1):
                    best_path[t] = backpointer[best_path[t+1], t+1]
                
                return [self.states[s] for s in best_path], best_path_prob
        
        # Create and train HMM
        states = ['easy', 'medium', 'hard']
        observations = ['easy', 'medium', 'hard']
        
        hmm = SimpleHMM(states, observations)
        
        # Train on all sequences
        all_sequences = list(difficulty_sequences.values())
        hmm.fit(all_sequences)
        
        print("HMM Parameters:")
        print(f"Start probabilities: {dict(zip(states, hmm.start_prob))}")
        print(f"\nTransition probabilities:")
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                print(f"  {from_state} -> {to_state}: {hmm.trans_prob[i, j]:.3f}")
        
        # Test prediction on sample sequences
        print(f"\nSequence Predictions:")
        for unit in sorted(list(difficulty_sequences.keys())[:3]):  # Test on first 3 units
            sequence = difficulty_sequences[unit]
            if len(sequence) > 2:
                # Predict on partial sequence
                partial_seq = sequence[:-1]
                predicted_path, prob = hmm.viterbi(partial_seq)
                
                print(f"\nUnit {unit} ({UNITS[unit]}):")
                print(f"  Actual sequence: {sequence}")
                print(f"  Predicted path: {predicted_path}")
                print(f"  Probability: {prob:.6f}")
                print(f"  Next predicted: {predicted_path[-1] if predicted_path else 'N/A'}")
                print(f"  Actual next: {sequence[-1]}")
        
        print("\nRESULT: HMM for sequential difficulty prediction implemented successfully!")
        return hmm, difficulty_sequences
    
    def experiment_8_cart_decision_tree(self):
        """
        EXPERIMENT 8: CART Decision Tree for Question Classification
        AIM: To implement CART learning algorithm for TNPSC question categorization
        """
        print("\n" + "="*50)
        print("EXPERIMENT 8: CART DECISION TREE")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare features from question text and metadata")
        print("2. Create decision tree classifier using CART algorithm")
        print("3. Train the model on TNPSC questions")
        print("4. Evaluate performance and visualize the tree")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features
        df = self.questions_df.copy()
        
        # Create comprehensive features
        df['question_length'] = df['question'].str.len()
        df['word_count'] = df['question'].str.split().str.len()
        df['has_numbers'] = df['question'].str.contains(r'\d').astype(int)
        df['has_punctuation'] = df['question'].str.contains(r'[^\w\s]').astype(int)
        df['avg_word_length'] = df['question'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        df['question_mark_count'] = df['question'].str.count(r'\?')
        df['capital_letters_count'] = df['question'].str.count(r'[A-Z]')
        
        # Text-based features using keyword matching
        science_keywords = ['physics', 'chemistry', 'biology', 'science', 'atom', 'cell', 'energy']
        history_keywords = ['history', 'ancient', 'medieval', 'empire', 'dynasty', 'culture']
        geography_keywords = ['geography', 'river', 'mountain', 'climate', 'ocean', 'continent']
        polity_keywords = ['constitution', 'government', 'parliament', 'president', 'democracy']
        economy_keywords = ['economy', 'economic', 'gdp', 'inflation', 'bank', 'finance']
        
        df['science_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in science_keywords if keyword in x))
        df['history_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in history_keywords if keyword in x))
        df['geography_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in geography_keywords if keyword in x))
        df['polity_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in polity_keywords if keyword in x))
        df['economy_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in economy_keywords if keyword in x))
        
        # Features and target
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers', 
                          'has_punctuation', 'avg_word_length', 'question_mark_count',
                          'capital_letters_count', 'science_keywords', 'history_keywords',
                          'geography_keywords', 'polity_keywords', 'economy_keywords']
        
        X = df[feature_columns]
        y = df['unit']  # Predict unit
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create and train decision tree
        dt_classifier = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        dt_classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dt_classifier.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Visualize decision tree
        plt.figure(figsize=(20, 12))
        from sklearn.tree import plot_tree
        
        plot_tree(dt_classifier, 
                 feature_names=feature_columns,
                 class_names=[f"Unit {i}" for i in sorted(y.unique())],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title('CART Decision Tree for TNPSC Question Classification')
        plt.show()
        
        # Show some sample predictions
        print(f"\nSample Predictions:")
        for i in range(min(5, len(y_test))):
            actual_unit = y_test.iloc[i]
            predicted_unit = y_pred[i]
            question_idx = y_test.index[i]
            question_text = df.loc[question_idx, 'question']
            
            print(f"\nQuestion: {question_text[:100]}...")
            print(f"Actual Unit: {actual_unit} ({UNITS[actual_unit]})")
            print(f"Predicted Unit: {predicted_unit} ({UNITS[predicted_unit]})")
            print(f"Correct: {'Yes' if actual_unit == predicted_unit else 'No'}")
        
        print("\nRESULT: CART decision tree classifier implemented and evaluated successfully!")
        return dt_classifier, accuracy, importance_df
    
    def experiment_9_ensemble_learning(self):
        """
        EXPERIMENT 9: Ensemble Learning for TNPSC Question Classification
        AIM: To implement ensemble learning models for improved classification performance
        """
        print("\n" + "="*50)
        print("EXPERIMENT 9: ENSEMBLE LEARNING")
        print("="*50)
        
        print("ALGORITHM:")
        print("1. Prepare features for ensemble learning")
        print("2. Implement Random Forest (Bagging)")
        print("3. Implement AdaBoost (Boosting)")
        print("4. Compare performance of ensemble methods")
        print("5. Analyze feature importance and model predictions")
        
        print("\nPROGRAM EXECUTION:")
        
        # Prepare features (same as decision tree)
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
        polity_keywords = ['constitution', 'government', 'parliament', 'president', 'democracy']
        economy_keywords = ['economy', 'economic', 'gdp', 'inflation', 'bank', 'finance']
        
        df['science_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in science_keywords if keyword in x))
        df['history_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in history_keywords if keyword in x))
        df['geography_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in geography_keywords if keyword in x))
        df['polity_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in polity_keywords if keyword in x))
        df['economy_keywords'] = df['question'].str.lower().apply(
            lambda x: sum(1 for keyword in economy_keywords if keyword in x))
        
        # Features and target
        feature_columns = ['year', 'question_length', 'word_count', 'has_numbers', 
                          'has_punctuation', 'avg_word_length', 'question_mark_count',
                          'capital_letters_count', 'science_keywords', 'history_keywords',
                          'geography_keywords', 'polity_keywords', 'economy_keywords']
        
        X = df[feature_columns]
        y = df['unit']  # Predict unit
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 1. Random Forest (Bagging)
        print("Training Random Forest...")
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        rf_classifier.fit(X_train, y_train)
        rf_pred = rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # 2. AdaBoost (Boosting)
        print("Training AdaBoost...")
        ada_classifier = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
        ada_classifier.fit(X_train, y_train)
        ada_pred = ada_classifier.predict(X_test)
        ada_accuracy = accuracy_score(y_test, ada_pred)
        
        # 3. Single Decision Tree for comparison
        print("Training Single Decision Tree...")
        dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt_classifier.fit(X_train, y_train)
        dt_pred = dt_classifier.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        # Compare results
        print(f"\nModel Performance Comparison:")
        print(f"Single Decision Tree Accuracy: {dt_accuracy:.4f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")
        
        # Detailed classification reports
        print(f"\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred))
        
        print(f"\nAdaBoost Classification Report:")
        print(classification_report(y_test, ada_pred))
        
        # Feature importance comparison
        rf_importance = rf_classifier.feature_importances_
        ada_importance = ada_classifier.feature_importances_
        
        importance_comparison = pd.DataFrame({
            'feature': feature_columns,
            'random_forest': rf_importance,
            'adaboost': ada_importance
        }).sort_values('random_forest', ascending=False)
        
        print(f"\nFeature Importance Comparison:")
        print(importance_comparison)
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Model accuracy comparison
        plt.subplot(2, 3, 1)
        models = ['Decision Tree', 'Random Forest', 'AdaBoost']
        accuracies = [dt_accuracy, rf_accuracy, ada_accuracy]
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Feature importance - Random Forest
        plt.subplot(2, 3, 2)
        top_features_rf = importance_comparison.head(8)
        plt.barh(range(len(top_features_rf)), top_features_rf['random_forest'])
        plt.yticks(range(len(top_features_rf)), top_features_rf['feature'])
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Importance')
        
        # Feature importance - AdaBoost
        plt.subplot(2, 3, 3)
        top_features_ada = importance_comparison.sort_values('adaboost', ascending=False).head(8)
        plt.barh(range(len(top_features_ada)), top_features_ada['adaboost'])
        plt.yticks(range(len(top_features_ada)), top_features_ada['feature'])
        plt.title('AdaBoost Feature Importance')
        plt.xlabel('Importance')
        
        # Prediction comparison for first few samples
        plt.subplot(2, 3, 4)
        sample_size = min(10, len(y_test))
        x_pos = np.arange(sample_size)
        width = 0.25
        
        plt.bar(x_pos - width, y_test.iloc[:sample_size], width, label='Actual', alpha=0.7)
        plt.bar(x_pos, rf_pred[:sample_size], width, label='Random Forest', alpha=0.7)
        plt.bar(x_pos + width, ada_pred[:sample_size], width, label='AdaBoost', alpha=0.7)
        
        plt.title('Sample Predictions Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('Unit')
        plt.legend()
        plt.xticks(x_pos, range(sample_size))
        
        # Confusion matrix for Random Forest
        plt.subplot(2, 3, 5)
        from sklearn.metrics import confusion_matrix
        cm_rf = confusion_matrix(y_test, rf_pred)
        plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Random Forest Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y)))
        plt.xticks(tick_marks, np.unique(y))
        plt.yticks(tick_marks, np.unique(y))
        plt.xlabel('Predicted Unit')
        plt.ylabel('Actual Unit')
        
        # Confusion matrix for AdaBoost
        plt.subplot(2, 3, 6)
        cm_ada = confusion_matrix(y_test, ada_pred)
        plt.imshow(cm_ada, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title('AdaBoost Confusion Matrix')
        plt.colorbar()
        plt.xticks(tick_marks, np.unique(y))
        plt.yticks(tick_marks, np.unique(y))
        plt.xlabel('Predicted Unit')
        plt.ylabel('Actual Unit')
        
        plt.tight_layout()
        plt.show()
        
        # Sample predictions analysis
        print(f"\nSample Predictions Analysis:")
        for i in range(min(5, len(y_test))):
            actual_unit = y_test.iloc[i]
            rf_pred_unit = rf_pred[i]
            ada_pred_unit = ada_pred[i]
            question_idx = y_test.index[i]
            question_text = df.loc[question_idx, 'question']
            
            print(f"\nSample {i+1}:")
            print(f"Question: {question_text[:80]}...")
            print(f"Actual Unit: {actual_unit} ({UNITS[actual_unit]})")
            print(f"Random Forest: {rf_pred_unit} ({UNITS[rf_pred_unit]})")
            print(f"AdaBoost: {ada_pred_unit} ({UNITS[ada_pred_unit]})")
            print(f"RF Correct: {'Yes' if actual_unit == rf_pred_unit else 'No'}")
            print(f"Ada Correct: {'Yes' if actual_unit == ada_pred_unit else 'No'}")
        
        print("\nRESULT: Ensemble learning models implemented and compared successfully!")
        return {
            'random_forest': (rf_classifier, rf_accuracy),
            'adaboost': (ada_classifier, ada_accuracy),
            'decision_tree': (dt_classifier, dt_accuracy),
            'feature_importance': importance_comparison
        }
    
    def run_all_experiments(self):
        """Run all 9 experiments in sequence"""
        print("="*60)
        print("RUNNING ALL TNPSC ML EXPERIMENTS")
        print("="*60)
        
        results = {}
        
        try:
            # Experiment 1
            results['exp1'] = self.experiment_1_load_and_view_dataset()
            
            # Experiment 2
            results['exp2'] = self.experiment_2_dataset_statistics()
            
            # Experiment 3
            results['exp3'] = self.experiment_3_linear_regression_prediction()
            
            # Experiment 4.1
            results['exp4_1'] = self.experiment_4_1_bayesian_logistic_regression()
            
            # Experiment 4.2
            results['exp4_2'] = self.experiment_4_2_svm_classification()
            
            # Experiment 5.1
            results['exp5_1'] = self.experiment_5_1_kmeans_clustering()
            
            # Experiment 5.2
            results['exp5_2'] = self.experiment_5_2_gaussian_mixture_models()
            
            # Experiment 5.3
            results['exp5_3'] = self.experiment_5_3_hierarchical_clustering()
            
            # Experiment 6
            results['exp6'] = self.experiment_6_pca_analysis()
            
            # Experiment 7
            results['exp7'] = self.experiment_7_hmm_sequential_prediction()
            
            # Experiment 8
            results['exp8'] = self.experiment_8_cart_decision_tree()
            
            # Experiment 9
            results['exp9'] = self.experiment_9_ensemble_learning()
            
            print("\n" + "="*60)
            print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"Error in experiments: {e}")
            import traceback
            traceback.print_exc()
        
        return results

# Main execution
if __name__ == "__main__":
    # Initialize experiments
    experiments = TNPSCMLExperiments()
    
    # Run individual experiment or all
    print("Choose an option:")
    print("1-9: Run individual experiment")
    print("0: Run all experiments")
    
    choice = input("Enter your choice (0-9): ").strip()
    
    if choice == "0":
        results = experiments.run_all_experiments()
    elif choice == "1":
        experiments.experiment_1_load_and_view_dataset()
    elif choice == "2":
        experiments.experiment_2_dataset_statistics()
    elif choice == "3":
        experiments.experiment_3_linear_regression_prediction()
    elif choice == "4":
        print("Choose 4.1 (Bayesian Logistic) or 4.2 (SVM):")
        sub_choice = input("Enter 1 or 2: ").strip()
        if sub_choice == "1":
            experiments.experiment_4_1_bayesian_logistic_regression()
        elif sub_choice == "2":
            experiments.experiment_4_2_svm_classification()
    elif choice == "5":
        print("Choose 5.1 (K-Means), 5.2 (GMM), or 5.3 (Hierarchical):")
        sub_choice = input("Enter 1, 2, or 3: ").strip()
        if sub_choice == "1":
            experiments.experiment_5_1_kmeans_clustering()
        elif sub_choice == "2":
            experiments.experiment_5_2_gaussian_mixture_models()
        elif sub_choice == "3":
            experiments.experiment_5_3_hierarchical_clustering()
    elif choice == "6":
        experiments.experiment_6_pca_analysis()
    elif choice == "7":
        experiments.experiment_7_hmm_sequential_prediction()
    elif choice == "8":
        experiments.experiment_8_cart_decision_tree()
    elif choice == "9":
        experiments.experiment_9_ensemble_learning()
    else:
        print("Invalid choice. Running all experiments...")
        results = experiments.run_all_experiments()