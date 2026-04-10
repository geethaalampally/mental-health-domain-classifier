# 🧠 Mental Health Domain Classification using NLP & Transformers

A comprehensive NLP + Deep Learning + Transformer-based project that classifies mental health-related questions into multiple domains such as Anxiety, Depression, Stress, and more.

This project provides a complete end-to-end pipeline — from data preprocessing to model training, evaluation, and comparison across multiple approaches.

---

## 📌 Overview

Mental health is a critical global concern. Automatically classifying mental health queries into specific domains can help in:

- Better organization of support systems  
- Improved chatbot responses  
- Faster assistance routing  

In this project, we classify mental health questions using:

- Machine Learning models  
- Deep Learning models  
- Transformer-based models  

---

## 🎯 Objectives

- Classify mental health questions into domains  
- Compare ML, DL, and Transformer models  
- Evaluate models using multiple performance metrics  
- Visualize model performance for better insights  

---

## 🧠 Domains Covered

- Anxiety  
- Depression  
- Stress  
- Anger  
- Loneliness  
- Motivation  
- Confusion  
*(based on dataset labels)*

---

## 🤖 Models Used

### 🔹 Machine Learning
- Logistic Regression  
- Naive Bayes  
- Random Forest  

### 🔹 Deep Learning
- LSTM  
- GRU  

### 🔹 Transformers
- BERT  
- DistilBERT  
- RoBERTa  

---

## 📊 Evaluation Metrics

We evaluated all models using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

Additional analysis includes:
- Confusion Matrix  
- ROC Curve  
- Model comparison charts  

---

## 📈 Visualizations

The project includes:

- Domain distribution (pie chart)  
- Word frequency bar charts  
- WordCloud visualization  
- Confusion matrices  
- ROC curves  
- Accuracy comparison graphs  
- Multi-metric comparison plots  

---

---

## ⚙️ Pipeline

1. Data Loading (MHQA datasets)  
2. Data Cleaning & Text Preprocessing  
3. Feature Engineering (TF-IDF & Tokenization)  
4. Model Training:
   - ML models
   - DL models
   - Transformer models  
5. Model Evaluation  
6. Visualization  
7. Final Model Comparison  

---

## 💡 Key Insights

- Transformer models (BERT, DistilBERT, RoBERTa) performed best overall  
- DistilBERT provided a strong balance between speed and accuracy  
- Traditional ML models (Logistic Regression) also performed competitively  
- Deep Learning models (LSTM, GRU) underperformed compared to Transformers  

---

## ⚠️ Challenges

- Dataset imbalance across domains  
- Training time for transformer models  
- Resource constraints (CPU/GPU limitations)  
- Hyperparameter tuning complexity  

---

## 🚀 Future Work

- Fine-tune models with larger datasets  
- Use domain-specific models (MentalBERT)  
- Apply hyperparameter optimization  
- Build real-time chatbot integration  
- Deploy using Streamlit / Flask API  

---

## 🧑‍💻 Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- HuggingFace Transformers  
- PyTorch  
- NLTK  
- Matplotlib & Seaborn  

---

## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to project
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
