#!/usr/bin/env python3
"""
TNPSC Model Training with Visualization
Generates: Training/Validation Accuracy, Loss, and Jaccard Coefficient graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, jaccard_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import core assistant
from tnpsc_assistant import TNPSCExamAssistant, UNITS

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.size'] = 10


class TNPSCModelTrainer:
    """Train TNPSC models with history tracking for visualization"""
    
    def __init__(self):
        """Initialize trainer"""
        print("🚀 Initializing TNPSC Model Trainer...")
        print("="*60)
        
        # Load data
        self.assistant = TNPSCExamAssistant()
        self.df = self.assistant.questions_df
        
        # Prepare features
        self._prepare_features()
        
        # Training history
        self.history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': [],
            'train_jaccard': [],
            'val_jaccard': []
        }
        
        print("✅ Trainer initialized!")
        print("="*60)
    
    def _prepare_features(self):
        """Prepare enhanced features"""
        print("🔧 Preparing features...")
        
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['has_numbers'] = self.df['question'].str.contains(r'\d').astype(int)
        self.df['has_question_mark'] = self.df['question'].str.contains(r'\?').astype(int)
        self.df['avg_word_length'] = self.df['question'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Features
        self.feature_cols = ['unit', 'year', 'question_length', 'word_count', 
                             'has_numbers', 'has_question_mark', 'avg_word_length']
        
        self.X = self.df[self.feature_cols]
        self.y = self.df['difficulty']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"   ✅ Features prepared: {self.X.shape}")
        print(f"   ✅ Classes: {list(self.label_encoder.classes_)}")
    
    def train_with_history(self, n_estimators_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        """
        Train model incrementally to track training history
        Simulates epochs by increasing number of estimators
        """
        print("\n🎯 Training model with history tracking...")
        print("="*60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print()
        
        # Train with increasing complexity (simulating epochs)
        for epoch, n_est in enumerate(n_estimators_list, 1):
            print(f"Epoch {epoch}/{len(n_estimators_list)} (n_estimators={n_est})...", end=" ")
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                warm_start=False
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Probabilities for loss calculation
            y_train_proba = model.predict_proba(X_train)
            y_val_proba = model.predict_proba(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            train_loss = log_loss(y_train, y_train_proba)
            val_loss = log_loss(y_val, y_val_proba)
            
            train_jaccard = jaccard_score(y_train, y_train_pred, average='weighted')
            val_jaccard = jaccard_score(y_val, y_val_pred, average='weighted')
            
            # Store history
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_jaccard'].append(train_jaccard)
            self.history['val_jaccard'].append(val_jaccard)
            
            print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        print("\n✅ Training complete!")
        print("="*60)
        
        # Store final model
        self.model = model
        
        return self.history
    
    def plot_training_history(self, output_dir='outputs'):
        """Generate three separate graph files"""
        print("\n📊 Generating training visualizations...")
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        epochs = range(1, len(self.history['train_accuracy']) + 1)
        
        saved_files = []
        
        # ========== GRAPH 1: Training and Validation Accuracy ==========
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_accuracy'], 'b-o', label='Training Accuracy', 
                 linewidth=2.5, markersize=8)
        plt.plot(epochs, self.history['val_accuracy'], 'r-s', label='Validation Accuracy', 
                 linewidth=2.5, markersize=8)
        plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 1.05])
        
        # Add value labels on points
        for i, (train_acc, val_acc) in enumerate(zip(self.history['train_accuracy'], 
                                                      self.history['val_accuracy'])):
            if i % 2 == 0:  # Show every other label
                plt.text(i+1, train_acc + 0.02, f'{train_acc:.3f}', ha='center', 
                        va='bottom', fontsize=9, color='blue')
                plt.text(i+1, val_acc - 0.02, f'{val_acc:.3f}', ha='center', 
                        va='top', fontsize=9, color='red')
        
        plt.tight_layout()
        accuracy_path = os.path.join(output_dir, 'training_validation_accuracy.png')
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graph 1 saved: {accuracy_path}")
        saved_files.append(accuracy_path)
        plt.close()
        
        # ========== GRAPH 2: Training and Validation Loss ==========
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Training Loss', 
                 linewidth=2.5, markersize=8)
        plt.plot(epochs, self.history['val_loss'], 'r-s', label='Validation Loss', 
                 linewidth=2.5, markersize=8)
        plt.title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Loss (Log Loss)', fontsize=13, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (train_loss, val_loss) in enumerate(zip(self.history['train_loss'], 
                                                        self.history['val_loss'])):
            if i % 2 == 0:
                plt.text(i+1, train_loss + 0.01, f'{train_loss:.3f}', ha='center', 
                        va='bottom', fontsize=9, color='blue')
                plt.text(i+1, val_loss - 0.01, f'{val_loss:.3f}', ha='center', 
                        va='top', fontsize=9, color='red')
        
        plt.tight_layout()
        loss_path = os.path.join(output_dir, 'training_validation_loss.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graph 2 saved: {loss_path}")
        saved_files.append(loss_path)
        plt.close()
        
        # ========== GRAPH 3: Training and Validation Jaccard Coefficient ==========
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_jaccard'], 'b-o', label='Training Jaccard', 
                 linewidth=2.5, markersize=8)
        plt.plot(epochs, self.history['val_jaccard'], 'r-s', label='Validation Jaccard', 
                 linewidth=2.5, markersize=8)
        plt.title('Training and Validation Jaccard Coefficient', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('Jaccard Coefficient', fontsize=13, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 1.05])
        
        # Add value labels
        for i, (train_jac, val_jac) in enumerate(zip(self.history['train_jaccard'], 
                                                      self.history['val_jaccard'])):
            if i % 2 == 0:
                plt.text(i+1, train_jac + 0.02, f'{train_jac:.3f}', ha='center', 
                        va='bottom', fontsize=9, color='blue')
                plt.text(i+1, val_jac - 0.02, f'{val_jac:.3f}', ha='center', 
                        va='top', fontsize=9, color='red')
        
        plt.tight_layout()
        jaccard_path = os.path.join(output_dir, 'training_validation_jaccard.png')
        plt.savefig(jaccard_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graph 3 saved: {jaccard_path}")
        saved_files.append(jaccard_path)
        plt.close()
        
        return saved_files
    
    def print_final_metrics(self):
        """Print final training metrics"""
        print("\n📈 FINAL METRICS:")
        print("="*60)
        print(f"Final Training Accuracy:   {self.history['train_accuracy'][-1]:.4f}")
        print(f"Final Validation Accuracy: {self.history['val_accuracy'][-1]:.4f}")
        print(f"Final Training Loss:       {self.history['train_loss'][-1]:.4f}")
        print(f"Final Validation Loss:     {self.history['val_loss'][-1]:.4f}")
        print(f"Final Training Jaccard:    {self.history['train_jaccard'][-1]:.4f}")
        print(f"Final Validation Jaccard:  {self.history['val_jaccard'][-1]:.4f}")
        print("="*60)
        
        # Check for overfitting
        acc_diff = self.history['train_accuracy'][-1] - self.history['val_accuracy'][-1]
        if acc_diff > 0.1:
            print("⚠️  Warning: Possible overfitting detected (train-val accuracy gap > 0.1)")
        else:
            print("✅ Model shows good generalization!")
        print()


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("TNPSC MODEL TRAINING & VISUALIZATION")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = TNPSCModelTrainer()
    
    # Train with history tracking
    history = trainer.train_with_history()
    
    # Plot training history (3 separate files)
    saved_files = trainer.plot_training_history()
    
    # Print final metrics
    trainer.print_final_metrics()
    
    print("🎉 Training visualization complete!")
    print("\n📊 Generated 3 separate graph files:")
    for i, file_path in enumerate(saved_files, 1):
        print(f"   {i}. {file_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
