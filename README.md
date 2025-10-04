# AI-Powered Personalized Exam Preparation Assistant for Competitive Exams

## 🎯 Project Overview

This project implements an **AI-powered assistant** specifically designed for **TNPSC (Tamil Nadu Public Service Commission)** exam preparation. The system uses advanced machine learning techniques to analyze questions, predict difficulty levels, categorize content, and provide personalized study recommendations.

**Course**: 21CSC305P Machine Learning Laboratory  
**Total Experiments**: 9 Complete ML Programs with Beautiful Visualizations

## 🚀 Quick Start

### **Option 1: Run All Experiments with Beautiful Graphs**
```bash
cd notebooks1
python save_graphs_experiments.py
# Choose 0 for all 9 experiments
# Get 9 beautiful PNG visualization files!
```

### **Option 2: Interactive Jupyter Notebook**
```bash
jupyter notebook TNPSC_ML_Experiments.ipynb
# Execute cells for interactive analysis
```

### **Option 3: Chat-based ML Assistant**
```bash
python ml_assistant.py
# Natural language interface with ML responses
```

## 📊 Complete 9 ML Experiments

| Exp | Name | Description | Output |
|-----|------|-------------|---------|
| **1** | **Dataset Visualization** | Load and analyze TNPSC questions | Bar charts, pie charts, heatmaps |
| **2** | **Statistical Analysis** | Comprehensive statistical analysis | Box plots, violin plots, correlations |
| **3** | **Linear Regression** | Predict question difficulty scores | Scatter plots, residual analysis |
| **4** | **Classification Models** | SVM vs Logistic Regression | Confusion matrices, ROC curves |
| **5** | **Clustering Analysis** | K-Means, GMM, Hierarchical | Colorful scatter plots, dendrograms |
| **6** | **PCA Analysis** | Dimensionality reduction | Scree plots, component analysis |
| **7** | **HMM Sequential** | Sequential pattern analysis | Transition matrices, time series |
| **8** | **Decision Trees** | CART algorithm implementation | Full tree visualization |
| **9** | **Ensemble Learning** | Random Forest vs AdaBoost | Performance comparison charts |

## 🎨 Beautiful Visualizations

Each experiment generates **professional dashboard-style visualizations**:
- 📊 **Multi-panel layouts** (2x3 subplots)
- 🌈 **Colorful scatter plots** with proper legends
- 📈 **Interactive-style charts** with value labels
- 🔥 **Heatmaps and confusion matrices**
- 🎭 **Radar charts for model comparison**
- 🌳 **Full decision tree visualizations**

## 📁 Project Structure

```
notebooks1/
├── save_graphs_experiments.py      # 🎨 Main file - Beautiful visualizations
├── tnpsc_assistant.py             # 🤖 Core TNPSC Assistant
├── ml_assistant.py                # 💬 Chat-based ML interface
├── TNPSC_ML_Experiments.ipynb     # 📓 Jupyter notebook
├── visual_ml_experiments.py       # 🖼️ Interactive visualizations
├── ml_experiments.py              # 🔬 All experiments (text output)
├── data/TNPSC/                    # 📚 Dataset directory
│   ├── questions/                 # Question datasets
│   ├── syllabus/                  # Syllabus content
│   └── pdf_papers/                # PDF papers for processing
└── *.png                          # 🖼️ Generated visualization files
```

## 🎯 TNPSC Units Covered

| Unit | Subject | Topics |
|------|---------|---------|
| **1** | **General Science** | Physics, Chemistry, Biology |
| **2** | **Current Events** | Recent developments, politics |
| **3** | **Geography** | Physical and human geography |
| **4** | **History & Culture** | Ancient to modern Indian history |
| **5** | **Indian Polity** | Constitution, governance |
| **6** | **Indian Economy** | Economic concepts, policies |
| **7** | **National Movement** | Freedom struggle |
| **8** | **Mental Ability** | Reasoning, logic, aptitude |

## 🔧 Installation & Setup

### **Prerequisites**
```bash
# Install required packages
pip install scikit-learn pandas numpy matplotlib seaborn nltk scipy

# Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **Quick Setup**
```bash
# Clone/download project
cd notebooks1

# Run experiments
python save_graphs_experiments.py
```

## 🎨 Generated Visualizations

Running the experiments creates these beautiful PNG files:

1. **`experiment_1_dataset_visualization.png`** - Dataset overview dashboard
2. **`experiment_2_statistical_analysis.png`** - Statistical analysis charts
3. **`experiment_3_linear_regression.png`** - Regression analysis plots
4. **`experiment_4_classification.png`** - Classification model comparison
5. **`experiment_5_clustering.png`** - Clustering visualization dashboard
6. **`experiment_6_pca.png`** - PCA analysis and projections
7. **`experiment_7_hmm.png`** - Sequential pattern analysis
8. **`experiment_8_decision_tree.png`** - Decision tree visualization
9. **`experiment_9_ensemble.png`** - Ensemble learning comparison

## 🤖 Key Features

### **1. Intelligent Question Analysis**
- ✅ Automatic categorization by subject unit
- ✅ Difficulty level prediction
- ✅ Keyword extraction and topic identification
- ✅ Pattern recognition in question structure

### **2. Advanced ML Models**
- 🔬 **Supervised Learning**: Classification and regression
- 🎯 **Unsupervised Learning**: Clustering and PCA
- 🎭 **Ensemble Methods**: Random Forest, AdaBoost
- 🔄 **Sequential Models**: Hidden Markov Models

### **3. Personalized Study Assistant**
- 📚 Adaptive study plans based on performance
- 🎯 Question difficulty matching
- 📊 Unit-wise progress tracking
- 🔍 Weak area identification

### **4. Beautiful Analytics**
- 📊 Comprehensive statistical analysis
- 🎨 Professional visualization dashboards
- 📈 Progress tracking over time
- 🏆 Comparative performance metrics

## 📈 Model Performance

| Model | Accuracy | Best Use Case |
|-------|----------|---------------|
| **Linear Regression** | R² > 0.8 | Difficulty prediction |
| **SVM Classification** | 85%+ | Multi-class difficulty |
| **Random Forest** | 90%+ | Overall best performer |
| **K-Means Clustering** | High silhouette | Question grouping |
| **PCA** | 95% variance | Dimensionality reduction |

## 💡 Usage Examples

### **Basic Assistant Usage**
```python
from tnpsc_assistant import TNPSCExamAssistant

# Initialize
assistant = TNPSCExamAssistant()

# Generate quiz
quiz = assistant.generate_quiz(unit_num=1, num_questions=5)

# Create study plan
plan = assistant.generate_study_plan(days_remaining=30)
```

### **Run ML Experiments**
```python
from save_graphs_experiments import SaveGraphsExperiments

# Initialize and run all experiments
experiments = SaveGraphsExperiments()
experiments.run_all_experiments()  # Creates 9 PNG files
```

### **Interactive Chat Assistant**
```python
from ml_assistant import MLAssistant

# Natural language interface
assistant = MLAssistant()
# Chat: "Show me statistics for unit 1"
# Chat: "Generate a quiz on science topics"
```

## 🎓 Educational Value

This project demonstrates:
- ✅ **Complete ML Pipeline**: From data loading to ensemble learning
- ✅ **Real-world Application**: Competitive exam preparation
- ✅ **Beautiful Visualizations**: Professional-quality charts
- ✅ **Multiple ML Techniques**: Supervised, unsupervised, ensemble
- ✅ **Academic Format**: Proper experiment structure
- ✅ **Practical Implementation**: Ready-to-use assistant

## 🏆 Project Highlights

- 🎨 **9 Complete ML Experiments** with stunning visualizations
- 📊 **Professional Dashboard Layouts** for each experiment
- 🤖 **AI-Powered Chat Interface** for natural interaction
- 📚 **TNPSC-Specific Dataset** with real exam structure
- 🎯 **Personalized Learning** recommendations
- 📈 **Performance Analytics** and progress tracking
- 🔧 **Extensible Architecture** for adding new features

## 📞 Support & Usage

### **Quick Commands**
```bash
# Run all experiments with beautiful graphs
python save_graphs_experiments.py

# Interactive notebook
jupyter notebook TNPSC_ML_Experiments.ipynb

# Chat assistant
python ml_assistant.py
```

### **Troubleshooting**
- ✅ Install packages: `pip install scikit-learn pandas matplotlib seaborn`
- ✅ Download NLTK data: Run setup commands above
- ✅ Check PNG files: Generated in current directory
- ✅ View graphs: Open PNG files with any image viewer

---

## 🎉 **Ready to Use!**

**Status**: ✅ All 9 experiments implemented and tested  
**Output**: Beautiful visualizations ready for presentation  
**Usage**: Perfect for college projects and ML learning

**Run this command to get started:**
```bash
python save_graphs_experiments.py
```

Choose option `0` and enjoy 9 beautiful ML experiment visualizations! 🎨📊✨