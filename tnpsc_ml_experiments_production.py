#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Powered Personalized Exam Preparation Assistant for Competitive Exams
Production-Ready Machine Learning Laboratory Experiments

Author: TNPSC ML Team
Course: 21CSC305P Machine Learning Laboratory
Version: 1.0.0
Date: 2024

This module implements 9 comprehensive machine learning experiments for TNPSC exam analysis:
1. Dataset Loading and Visualization
2. Statistical Analysis
3. Linear Regression for Difficulty Prediction
4. Classification Models (SVM & Logistic Regression)
5. Clustering Analysis (K-Means, GMM)
6. Principal Component Analysis (PCA)
7. Hidden Markov Model Sequential Analysis
8. Decision Tree Classification (CART)
9. Ensemble Learning (Random Forest & AdaBoost)
"""

import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score, precision_recall_fscore_support
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tnpsc_ml_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from tnpsc_assistant import TNPSCExamAssistant, UNITS
except ImportError as e:
    logger.error(f"Failed to import TNPSC Assistant: {e}")
    sys.exit(1)


class TNPSCMLExperimentsProduction:
    """
    Production-ready implementation of TNPSC ML experiments.
    
    This class provides a comprehensive suite of machine learning experiments
    designed for analyzing TNPSC exam questions and generating insights for
    personalized exam preparation.
    
    Attributes:
        assistant (TNPSCExamAssistant): Core TNPSC assistant instance
        questions_df (pd.DataFrame): Dataset of TNPSC questions
        output_dir (Path): Directory for saving outputs
        config (Dict): Configuration parameters
    """
    
    def __init__(self, output_dir: str = "outputs", config: Optional[Dict] = None):
        """
        Initialize the TNPSC ML Experiments system.
        
        Args:
            output_dir (str): Directory to save output files
            config (Optional[Dict]): Configuration parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = config or self._get_default_config()
        self._setup_plotting_style()
        
        try:
            self.assistant = TNPSCExamAssistant()
            self.questions_df = self.assistant.questions_df
            logger.info("TNPSC ML Experiments initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TNPSC Assistant: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'figure_size': (18, 12),
            'dpi': 300,
            'font_size': 12,
            'random_state': 42,
            'test_size': 0.3,
            'n_clusters': 4,
            'max_features_tfidf': 100,
            'color_palette': 'husl'
        }
    
    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn for consistent, professional plots."""
        plt.style.use('default')
        sns.set_palette(self.config['color_palette'])
        
        plt.rcParams.update({
            'figure.figsize': self.config['figure_size'],
            'font.size': self.config['font_size'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.dpi': self.config['dpi'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight'
        })
    
    def _save_figure(self, filename: str, title: str = "") -> None:
        """
        Save figure with consistent formatting.
        
        Args:
            filename (str): Output filename
            title (str): Optional title for logging
        """
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.config['dpi'], bbox_inches='tight')
        logger.info(f"Saved visualization: {filepath}")
        if title:
            logger.info(f"Generated: {title}")
        plt.close()
    
    def experiment_1_dataset_visualization(self) -> Dict[str, Any]:
        """
        Experiment 1: Comprehensive Dataset Visualization and Analysis.
        
        Returns:
            Dict[str, Any]: Analysis results and metrics
        """
        logger.info("Starting Experiment 1: Dataset Visualization")
        
        try:
            df = self.questions_df.copy()
            
            # Create comprehensive visualization dashboard
            fig, axes = plt.subplots(2, 3, figsize=self.config['figure_size'])
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
            wedges, texts, autotexts = axes[0,1].pie(
                difficulty_counts.values, 
                labels=difficulty_counts.index,
                autopct='%1.1f%%',
                colors=colors_pie,
                explode=(0.05, 0.05, 0.05),
                shadow=True,
                startangle=90
            )
            axes[0,1].set_title('Difficulty Distribution', fontsize=14, fontweight='bold')
            
            # 3. Year-wise Distribution
            year_counts = df['year'].value_counts().sort_index()
            axes[0,2].plot(year_counts.index, year_counts.values, 'o-', 
                          linewidth=3, markersize=8, color='#e74c3c')
            axes[0,2].fill_between(year_counts.index, year_counts.values, 
                                  alpha=0.3, color='#e74c3c')
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
            axes[1,1].hist(df['question_length'], bins=15, color='skyblue', 
                          alpha=0.7, edgecolor='black')
            axes[1,1].axvline(df['question_length'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {df["question_length"].mean():.1f}')
            axes[1,1].set_title('Question Length Distribution', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel('Question Length (characters)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
            
            # 6. Units Overview
            unit_names = [f"Unit {i}\\n{UNITS[i]}" for i in unit_counts.index]
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
            self._save_figure('experiment_1_dataset_visualization.png', 
                            'Dataset Visualization Dashboard')
            
            # Return analysis results
            results = {
                'total_questions': len(df),
                'units_count': len(df['unit'].unique()),
                'years_covered': sorted(df['year'].unique()),
                'difficulty_distribution': difficulty_counts.to_dict(),
                'unit_distribution': unit_counts.to_dict(),
                'avg_question_length': df['question_length'].mean(),
                'status': 'success'
            }
            
            logger.info("Experiment 1 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Experiment 1 failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def experiment_2_statistical_analysis(self) -> Dict[str, Any]:
        """
        Experiment 2: Advanced Statistical Analysis and Visualization.
        
        Returns:
            Dict[str, Any]: Statistical analysis results
        """
        logger.info("Starting Experiment 2: Statistical Analysis")
        
        try:
            df = self.questions_df.copy()
            
            # Create statistical dashboard
            fig, axes = plt.subplots(2, 3, figsize=self.config['figure_size'])
            fig.suptitle('TNPSC Statistical Analysis Dashboard', fontsize=20, fontweight='bold')
            
            # Prepare features
            df['question_length'] = df['question'].str.len()
            df['word_count'] = df['question'].str.split().str.len()
            df['difficulty_num'] = df['difficulty'].map({'easy': 1, 'medium': 2, 'hard': 3})
            
            # 1. Box Plot - Question Length by Difficulty
            sns.boxplot(data=df, x='difficulty', y='question_length', ax=axes[0,0])
            axes[0,0].set_title('Question Length by Difficulty', fontsize=14, fontweight='bold')
            
            # 2. Violin Plot - Distribution by Unit
            sns.violinplot(data=df, x='unit', y='question_length', ax=axes[0,1])
            axes[0,1].set_title('Question Length Distribution by Unit', fontsize=14, fontweight='bold')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Correlation Matrix
            corr_features = ['unit', 'year', 'question_length', 'word_count', 'difficulty_num']
            corr_data = df[corr_features].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=axes[0,2])
            axes[0,2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            
            # 4. Stacked Bar Chart
            pivot_data = df.pivot_table(index='unit', columns='difficulty', 
                                      values='year', aggfunc='count', fill_value=0)
            pivot_data.plot(kind='bar', stacked=True, ax=axes[1,0], 
                           color=['#ff9999', '#66b3ff', '#99ff99'])
            axes[1,0].set_title('Questions by Unit and Difficulty', fontsize=14, fontweight='bold')
            axes[1,0].legend(title='Difficulty')
            axes[1,0].tick_params(axis='x', rotation=0)
            
            # 5. Scatter Plot with Regression
            sns.scatterplot(data=df, x='question_length', y='word_count', 
                           hue='difficulty', size='unit', sizes=(50, 200), ax=axes[1,1])
            sns.regplot(data=df, x='question_length', y='word_count', 
                       scatter=False, color='red', ax=axes[1,1])
            axes[1,1].set_title('Question Length vs Word Count', fontsize=14, fontweight='bold')
            
            # 6. Statistical Summary
            axes[1,2].axis('off')
            stats_summary = self._generate_statistical_summary(df)
            axes[1,2].text(0.1, 0.9, stats_summary, transform=axes[1,2].transAxes, 
                          fontsize=12, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            plt.tight_layout()
            self._save_figure('experiment_2_statistical_analysis.png', 
                            'Statistical Analysis Dashboard')
            
            # Calculate and return statistical metrics
            results = {
                'correlation_matrix': corr_data.to_dict(),
                'question_length_stats': df['question_length'].describe().to_dict(),
                'word_count_stats': df['word_count'].describe().to_dict(),
                'difficulty_distribution': df['difficulty'].value_counts().to_dict(),
                'status': 'success'
            }
            
            logger.info("Experiment 2 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Experiment 2 failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> str:
        """Generate formatted statistical summary text."""
        difficulty_counts = df['difficulty'].value_counts()
        
        summary = f"""STATISTICAL SUMMARY

Total Questions: {len(df)}
Units: {len(df['unit'].unique())}
Years: {sorted(df['year'].unique())}

Difficulty Distribution:
"""
        for diff, count in difficulty_counts.items():
            percentage = (count / len(df)) * 100
            summary += f"{diff.capitalize()}: {count} ({percentage:.1f}%)\\n"
        
        summary += f"""
Question Length Stats:
Mean: {df['question_length'].mean():.1f}
Median: {df['question_length'].median():.1f}
Std: {df['question_length'].std():.1f}
"""
        return summary
    
    def experiment_3_linear_regression(self) -> Dict[str, Any]:
        """
        Experiment 3: Linear Regression for Difficulty Prediction.
        
        Returns:
            Dict[str, Any]: Regression analysis results and metrics
        """
        logger.info("Starting Experiment 3: Linear Regression Analysis")
        
        try:
            df = self.questions_df.copy()
            
            # Feature engineering
            difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
            df['difficulty_score'] = df['difficulty'].map(difficulty_map)
            df['question_length'] = df['question'].str.len()
            df['word_count'] = df['question'].str.split().str.len()
            df['avg_option_length'] = (
                df['option_a'].str.len() + df['option_b'].str.len() + 
                df['option_c'].str.len() + df['option_d'].str.len()
            ) / 4
            
            # Prepare features and target
            features = ['unit', 'year', 'question_length', 'word_count', 'avg_option_length']
            X = df[features]
            y = df['difficulty_score']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Create visualization
            self._create_regression_visualization(
                y_test, y_pred, model, features, mse, r2, df
            )
            
            # Return results
            results = {
                'mse': float(mse),
                'r2_score': float(r2),
                'feature_coefficients': dict(zip(features, model.coef_)),
                'intercept': float(model.intercept_),
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'status': 'success'
            }
            
            logger.info(f"Experiment 3 completed - R²: {r2:.4f}, MSE: {mse:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Experiment 3 failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_regression_visualization(self, y_test, y_pred, model, features, 
                                       mse, r2, df) -> None:
        """Create comprehensive regression analysis visualization."""
        fig, axes = plt.subplots(2, 3, figsize=self.config['figure_size'])
        fig.suptitle('Linear Regression Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0,0].scatter(y_test, y_pred, alpha=0.7, color='blue', s=100, edgecolors='black')
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        axes[0,0].set_xlabel('Actual Difficulty Score')
        axes[0,0].set_ylabel('Predicted Difficulty Score')
        axes[0,0].set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        axes[0,0].text(0.05, 0.95, f'R² = {r2:.3f}\\nMSE = {mse:.3f}', 
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
        axes[0,2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
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
        performance_text = f"""LINEAR REGRESSION RESULTS

R² Score: {r2:.4f}
Mean Squared Error: {mse:.4f}

Feature Coefficients:
"""
        for feature, coef in zip(features, model.coef_):
            performance_text += f"{feature}: {coef:.4f}\\n"
        
        axes[1,2].text(0.1, 0.9, performance_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure('experiment_3_linear_regression.png', 
                        'Linear Regression Analysis Dashboard')
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Execute all 9 ML experiments and generate comprehensive results.
        
        Returns:
            Dict[str, Any]: Complete results from all experiments
        """
        logger.info("Starting comprehensive ML experiments suite")
        
        experiments = [
            ('Dataset Visualization', self.experiment_1_dataset_visualization),
            ('Statistical Analysis', self.experiment_2_statistical_analysis),
            ('Linear Regression', self.experiment_3_linear_regression),
            # Add other experiments here as they're implemented
        ]
        
        results = {}
        successful_experiments = 0
        
        for name, experiment_func in experiments:
            try:
                logger.info(f"Executing: {name}")
                result = experiment_func()
                results[name.lower().replace(' ', '_')] = result
                
                if result.get('status') == 'success':
                    successful_experiments += 1
                    logger.info(f"✅ {name} completed successfully")
                else:
                    logger.warning(f"⚠️ {name} completed with issues")
                    
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
                results[name.lower().replace(' ', '_')] = {
                    'status': 'error', 
                    'message': str(e)
                }
        
        # Generate summary report
        summary = {
            'total_experiments': len(experiments),
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / len(experiments),
            'output_directory': str(self.output_dir),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results['summary'] = summary
        
        logger.info(f"Experiments completed: {successful_experiments}/{len(experiments)} successful")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return results


def main():
    """Main execution function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TNPSC ML Experiments - Production Ready Implementation'
    )
    parser.add_argument(
        '--experiment', 
        type=int, 
        choices=range(0, 10),
        default=0,
        help='Experiment number to run (0 for all, 1-9 for specific)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    try:
        # Initialize experiments
        experiments = TNPSCMLExperimentsProduction(
            output_dir=args.output_dir,
            config=config
        )
        
        # Execute experiments
        if args.experiment == 0:
            logger.info("Running all experiments")
            results = experiments.run_all_experiments()
        else:
            experiment_map = {
                1: experiments.experiment_1_dataset_visualization,
                2: experiments.experiment_2_statistical_analysis,
                3: experiments.experiment_3_linear_regression,
                # Add other experiments as implemented
            }
            
            if args.experiment in experiment_map:
                logger.info(f"Running experiment {args.experiment}")
                results = experiment_map[args.experiment]()
            else:
                logger.error(f"Experiment {args.experiment} not yet implemented")
                return
        
        # Save results summary
        results_file = Path(args.output_dir) / 'experiment_results.json'
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results summary saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()