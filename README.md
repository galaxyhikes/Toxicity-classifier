# Toxicity Classifier

A comprehensive machine learning project for detecting and classifying toxic comments across multiple toxicity categories and severity levels. Starting with a traditional ML-based classifier, this project was extended to explore advanced AI-powered classification using Google's Gemini API.

## Table of Contents

- [What It Does](#what-it-does)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Results & Evaluation](#results--evaluation)
- [Support](#support)

## What It Does

The Toxicity Classifier is designed to automatically detect and classify toxic comments in online discussions. It analyzes text and determines:

1. **Toxicity Level**: Assigns severity levels (Not Toxic, Mild, Moderate, or Severe)
2. **Toxicity Categories**: Identifies specific types of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate)
3. **Toxicity Score**: Provides a numerical confidence score between 0.0 (not toxic) and 1.0 (highly toxic)

This tool helps content moderation teams and platform developers maintain healthier online communities.

### Dataset

The project is built on **Kaggle's Jigsaw Toxic Comment Classification Challenge** dataset, comprising **159,450 comments** extracted from Wikipedia's talk pages. These comments are labeled across 6 toxicity categories. For efficient model exploration, the dataset was scaled to **50,000 comments** with **proportional representation of each category** to prevent model bias.

## Key Features

### Model-Based Classification (`toxicity classifier model.ipynb`)
- **Multiple ML Algorithms**: Implements and compares:
  - Naive Bayes (NB)
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
  - Random Forest (RF)
  - Gradient Boosting (GB)
- **Smart Data Preprocessing**: 
  - Text cleaning and normalization
  - Tokenization and lemmatization
  - Stop word removal
  - Sentiment analysis using VADER (scores range from -1 to 1)
- **Toxicity Severity Binning**: Comments reclassified into 4 bins:
  - **Non-Toxic**: No toxicity labels
  - **Mild**: 1 toxicity label
  - **Moderate**: 2 toxicity labels
  - **Severe**: 3+ toxicity labels
- **Feature Engineering**:
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
  - Count vectorization
  - Combined 20,000-dimensional feature extraction
- **Hyperparameter Optimization**: Randomized and Grid Search CV for tuning learning rates, depth constraints, feature selection, and regularization
- **Model Evaluation**: Classification reports, confusion matrices, ROC-AUC scores, and cross-validation

### AI-Powered Classification (`toxicity classifier with gemini api.ipynb`)
Built upon the core ML model, this extension explores whether advanced LLM reasoning can improve toxicity detection:
- **Google Gemini 2.0 Integration**: Leverages advanced AI reasoning for nuanced toxicity detection
- **Context-Aware Analysis**: Understands intent, tone, and subtle toxicity (e.g., sarcasm, passive-aggression)
- **Batch Processing**: Handles multiple comments efficiently with rate limiting
- **Comparative Approach**: Applies the same dataset and preprocessing to compare LLM-based vs. traditional ML classification
- **JSON Output Structure**: Extracts structured toxicity assessments from model responses for comparison with ML models

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/galaxyhikes/Toxicity-classifier.git
   cd Toxicity-classifier
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud scipy textblob google-generativeai nest-asyncio
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

4. **Prepare your dataset**:
   - Place your training data as `train.csv` in the project directory
   - Expected columns: `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

### Setup for Gemini API

If using the AI-powered classifier:

1. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Update the API key in the notebook:
   ```python
   genai.configure(api_key="YOUR_API_KEY_HERE")
   ```

## Project Structure

```
Toxicity-classifier/
├── README.md                                      # This file
├── toxicity classifier model.ipynb                # ML-based classifier notebook comprising of preprocessing steps and model training across multiple algorithms with hyperparameter tuning and evaluation
├── toxicity classifier with gemini api.ipynb      # AI-powered classifier notebook
├── Final Report.pdf                               # Project report
└── Text Analytics group ppt.pdf                   # Presentation slides
```

## How to Use

### 1. Core Model-Based Classification

**Start here:** Open `toxicity classifier model.ipynb` and follow these steps:

```python
# The notebook handles the full pipeline:
# 1. Load and explore data
# 2. Preprocess text (cleaning, tokenization, lemmatization)
# 3. Extract features (TF-IDF + Count Vectorization)
# 4. Train multiple models with hyperparameter tuning
# 5. Evaluate performance with cross-validation
# 6. Compare model performance metrics
```

**Example workflow**:
- Run cells sequentially to train classifiers
- Observe model comparison results showing accuracy, precision, recall, and AUC scores
- Visualizations include confusion matrices, ROC curves, and class distributions

### 2. Extended: AI-Powered Classification with Gemini

**After exploring the core model:** Open `toxicity classifier with gemini api.ipynb` to compare LLM-based classification:

```python
# The extension includes:
# 1. Data preparation and exploration (same dataset as core model)
# 2. Text preprocessing (consistent with ML approach)
# 3. Async batch processing with Gemini API
# 4. Structured toxicity assessment using LLM reasoning
# 5. Comparative analysis against traditional ML results
```

**Key features of the extension**:
- Batch processing with configurable size (default: 15 samples per batch)
- Rate limiting (60-second delays between batches)
- Robust JSON extraction from model responses
- Detailed toxicity categorization using advanced AI reasoning
- Comparative analysis with traditional ML model outputs

### Example: Classifying a Comment

```python
# Both notebooks process comments through their respective pipelines
# Output includes:
# - Toxicity Level: Not Toxic / Mild / Moderate / Severe
# - Toxicity Categories: Specific types detected
# - Toxicity Score: 0.0 - 1.0 confidence metric
# - Sentiment Analysis: Positive / Neutral / Negative
```

## Results & Evaluation

### Traditional ML Model Performance

The project evaluated six machine learning algorithms on both controlled training data and real-world external validation:

#### Training Performance
- **Multi-Layer Perceptron (MLP)**: 95% accuracy, 94% AUC-ROC
  - 71% recall and 77% precision for toxic comments
  - Strong balance between true positives and false positives
  - Significant gap between training and validation accuracy (overfitting concerns)

- **Gradient Boosting & Random Forest**: Best overall performers
  - Superior generalization to unseen data
  - More stable learning curves compared to MLP

- **Naive Bayes**: High precision but low recall
  - Excellent at avoiding false positives
  - Missed substantial portion of actual toxic comments

#### External Validation (Real-World Data)

Models were tested on **200 manually labeled Reddit comments** from r/politics to assess real-world robustness:

- **Support Vector Machine (SVM)**: F1 score **0.72**, AUC-ROC **0.79**
  - Best external validation performance
  - Maintained balanced precision-recall trade-off on unseen data
  - Superior generalization capability

- **MLP**: Struggled with external validation despite strong training metrics
  - Revealed overfitting limitations of models trained on controlled environments
  - Highlighted importance of testing on diverse, real-world datasets

### Key Insight

**Sentiment analysis alone is insufficient** for toxicity detection—toxicity is often masked within seemingly neutral linguistic structures. Word frequency distributions and advanced classification methodology are critical for accurate detection.

### AI-Based Extension: Gemini 2.0 Flash

Built upon core ML findings, this extension explores whether advanced LLM reasoning can improve toxicity detection:
- **Contextual Understanding**: Detects subtle toxicity (sarcasm, passive-aggression)
- **Comparative Analysis**: Identifies patterns that traditional ML may miss
- **Output Format**: Structured JSON with detailed reasoning and classifications

## Support

### Documentation
- Refer to the comments within each notebook for detailed explanations
- Check `Final Report.pdf` for comprehensive project methodology and results
- Review `Text Analytics group ppt.pdf` for visual presentation of findings

### Issues
If you encounter any issues:
- Check the notebook cells for error messages and NLTK/library download failures
- Ensure all dependencies are installed: `pip install --upgrade google-generativeai`
- For Gemini API errors, verify your API key and rate limits

### References
- [scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [TF-IDF Vectorization Guide](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

## License

This project is provided as-is for educational and research purposes. See the repository's LICENSE file for details.

This project was developed as part of a comprehensive text analytics study comparing traditional machine learning with modern AI approaches for content moderation.
