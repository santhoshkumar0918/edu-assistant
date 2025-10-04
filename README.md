# AI-Powered Personalized Exam Preparation Assistant for Competitive Exams

## Project Overview

This project implements an AI-powered assistant specifically designed for TNPSC (Tamil Nadu Public Service Commission) exam preparation. The system uses machine learning techniques to analyze questions, predict difficulty levels, categorize content, and provide personalized study recommendations.

## 🎯 Project Objectives

- **Personalized Learning**: Adapt to individual learning styles and progress
- **Intelligent Question Analysis**: Automatically categorize and analyze exam questions
- **Predictive Modeling**: Predict question difficulty and exam patterns
- **Study Plan Generation**: Create customized study schedules
- **Performance Analytics**: Track and analyze student performance

## 📁 Project Structure

```
notebooks1/
├── tnpsc_assistant.py          # Main TNPSC Assistant implementation
├── ml_assistant.py             # ML-powered chat assistant
├── ml_experiments.py           # All 9 ML experiments in Python
├── TNPSC_ML_Experiments.ipynb  # Jupyter notebook with experiments
├── data/                       # Dataset directory
│   └── TNPSC/
│       ├── questions/          # Question datasets
│       ├── syllabus/          # Syllabus content
│       └── pdf_papers/        # PDF papers for processing
├── exppdffintext.md           # Experiment specifications
└── README.md                  # This file
```

## 🧪 Machine Learning Experiments (9 Programs)

### Experiment 1: Dataset Loading and Analysis
- **AIM**: Load and analyze the TNPSC questions dataset
- **Features**: Data exploration, statistics, visualization
- **Output**: Dataset overview, unit distribution, question analysis

### Experiment 2: Statistical Analysis
- **AIM**: Comprehensive statistical analysis of the dataset
- **Features**: Descriptive statistics, distributions, correlations
- **Output**: Difficulty distribution, year-wise analysis, unit statistics

### Experiment 3: Linear Regression
- **AIM**: Predict question difficulty scores using regression
- **Features**: Question length, metadata, text features
- **Output**: Difficulty prediction model, performance metrics

### Experiment 4.1: Bayesian Logistic Regression
- **AIM**: Classify questions using Bayesian approach
- **Features**: TF-IDF text features, binary classification
- **Output**: Science vs Non-Science classification model

### Experiment 4.2: SVM Classification
- **AIM**: Multi-class classification using Support Vector Machines
- **Features**: Text and metadata features
- **Output**: Difficulty level classification model

### Experiment 5.1: K-Means Clustering
- **AIM**: Unsupervised clustering of questions
- **Features**: TF-IDF vectors, numerical features
- **Output**: Question clusters, cluster analysis

### Experiment 5.2: Gaussian Mixture Models
- **AIM**: Probabilistic clustering using GMM
- **Features**: Same as K-Means with probability assignments
- **Output**: Soft clustering results, model selection metrics

### Experiment 5.3: Hierarchical Clustering
- **AIM**: Hierarchical clustering analysis
- **Features**: Distance-based clustering
- **Output**: Dendrogram, hierarchical clusters

### Experiment 6: Principal Component Analysis
- **AIM**: Dimensionality reduction and feature analysis
- **Features**: High-dimensional feature vectors
- **Output**: Reduced dimensions, explained variance, visualizations

### Experiment 7: Hidden Markov Models
- **AIM**: Sequential pattern analysis in question difficulty
- **Features**: Difficulty sequences by unit and year
- **Output**: HMM parameters, sequence predictions

### Experiment 8: CART Decision Trees
- **AIM**: Decision tree classification using CART algorithm
- **Features**: Comprehensive question features
- **Output**: Decision tree model, feature importance

### Experiment 9: Ensemble Learning
- **AIM**: Combine multiple models for improved performance
- **Features**: Random Forest (Bagging) and AdaBoost (Boosting)
- **Output**: Ensemble model comparison, performance analysis

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tqdm
pip install jupyter notebook  # For running Jupyter notebooks
pip install PyPDF2  # For PDF processing (optional)
```

### Installation

1. Clone or download the project files
2. Install required dependencies
3. Run the setup to download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Running the Experiments

#### Option 1: Run Individual Experiments (Python)
```bash
python ml_experiments.py
# Choose option 0 for all experiments or 1-9 for individual experiments
```

#### Option 2: Run Jupyter Notebook
```bash
jupyter notebook TNPSC_ML_Experiments.ipynb
# Execute cells sequentially for interactive analysis
```

#### Option 3: Use the Interactive Assistant
```bash
python ml_assistant.py
# Chat-based interface with ML-powered responses
```

## 📊 Dataset Information

### TNPSC Units Covered:
1. **General Science** - Physics, Chemistry, Biology
2. **Current Events** - Recent developments, politics
3. **Geography** - Physical and human geography
4. **History and Culture of India** - Ancient to modern history
5. **Indian Polity** - Constitution, governance
6. **Indian Economy** - Economic concepts, policies
7. **Indian National Movement** - Freedom struggle
8. **Mental Ability & Aptitude** - Reasoning, logic

### Question Features:
- **Text Content**: Question text and multiple choice options
- **Metadata**: Unit, difficulty level, year
- **Derived Features**: Length, word count, keyword presence
- **Target Variables**: Unit classification, difficulty prediction

## 🔧 Key Features

### 1. Intelligent Question Analysis
- Automatic categorization by subject unit
- Difficulty level prediction
- Keyword extraction and topic identification
- Pattern recognition in question structure

### 2. Machine Learning Models
- **Supervised Learning**: Classification and regression models
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Ensemble Methods**: Random Forest, AdaBoost
- **Sequential Models**: Hidden Markov Models

### 3. Personalized Recommendations
- Adaptive study plans based on performance
- Question difficulty matching
- Unit-wise progress tracking
- Weak area identification

### 4. Performance Analytics
- Comprehensive statistical analysis
- Visualization of learning patterns
- Progress tracking over time
- Comparative performance metrics

## 📈 Results and Performance

### Model Performance Summary:
- **Linear Regression**: R² score for difficulty prediction
- **SVM Classification**: Multi-class accuracy for difficulty levels
- **Random Forest**: Best overall classification performance
- **K-Means Clustering**: Effective question grouping
- **PCA**: Significant dimensionality reduction with retained variance

### Key Insights:
- Text-based features are highly predictive of question categories
- Ensemble methods outperform individual classifiers
- Question length and complexity correlate with difficulty
- Different units have distinct linguistic patterns
- Sequential patterns exist in question difficulty progression

## 🛠️ Customization and Extension

### Adding New Features:
1. Modify feature extraction in `ml_experiments.py`
2. Add new algorithms in the experiments class
3. Update visualization functions for new metrics

### Dataset Expansion:
1. Add new questions to `data/TNPSC/questions/`
2. Include additional metadata fields
3. Process PDF papers using built-in PDF parser

### Model Tuning:
1. Adjust hyperparameters in each experiment
2. Implement cross-validation for better evaluation
3. Add new evaluation metrics as needed

## 📝 Usage Examples

### Basic Usage:
```python
from tnpsc_assistant import TNPSCExamAssistant

# Initialize assistant
assistant = TNPSCExamAssistant()

# Generate quiz
quiz = assistant.generate_quiz(unit_num=1, num_questions=5)

# Create study plan
plan = assistant.generate_study_plan(days_remaining=30)

# Get unit information
info = assistant.get_unit_info(unit_num=1)
```

### ML Experiments:
```python
from ml_experiments import TNPSCMLExperiments

# Initialize experiments
experiments = TNPSCMLExperiments()

# Run specific experiment
results = experiments.experiment_3_linear_regression_prediction()

# Run all experiments
all_results = experiments.run_all_experiments()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new experiments or improve existing ones
4. Test thoroughly with sample data
5. Submit a pull request

## 📄 License

This project is developed for educational purposes as part of the Machine Learning Laboratory course (21CSC305P).

## 🙏 Acknowledgments

- Tamil Nadu Public Service Commission for exam structure reference
- Scikit-learn community for ML algorithms
- NLTK team for natural language processing tools
- Matplotlib and Seaborn for visualization capabilities

## 📞 Support

For questions or issues:
1. Check the experiment outputs for debugging information
2. Verify all dependencies are installed correctly
3. Ensure dataset files are in the correct directory structure
4. Review the Jupyter notebook for step-by-step execution

---

**Project**: AI-Powered Personalized Exam Preparation Assistant for Competitive Exams  
**Course**: 21CSC305P Machine Learning Laboratory  
**Experiments**: 9 Complete ML Programs for TNPSC Exam Analysis

**Status**: ✅ All experiments implemented and tested successfully!