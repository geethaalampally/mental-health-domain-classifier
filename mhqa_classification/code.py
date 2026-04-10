# ============================================
# 1. FILE UPLOAD (COLAB)
# ============================================
from google.colab import files
uploaded = files.upload()

import os
os.listdir()


# ============================================
# 2. LOAD DATA
# ============================================
import pandas as pd

df1 = pd.read_csv("mhqa.csv")
df2 = pd.read_csv("mhqa-b.csv")

df = pd.concat([df1, df2], ignore_index=True)

print("df1:", df1.shape)
print("df2:", df2.shape)
print("Total:", df.shape)


# ============================================
# 3. PREPROCESSING
# ============================================
df = df[['question', 'topic']]
df.rename(columns={'topic': 'domain'}, inplace=True)
df.dropna(inplace=True)


# ============================================
# 4. TEXT CLEANING
# ============================================
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['question'].apply(clean_text)


# ============================================
# 5. TF-IDF (ML)
# ============================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['domain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ============================================
# 6. ML MODELS
# ============================================
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=200)

lr.fit(X_train, y_train)
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_nb = nb.predict(X_test)
y_pred_rf = rf.predict(X_test)


# ============================================
# 7. ML METRICS
# ============================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results_ml = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr, average='weighted'),
        precision_score(y_test, y_pred_nb, average='weighted'),
        precision_score(y_test, y_pred_rf, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr, average='weighted'),
        recall_score(y_test, y_pred_nb, average='weighted'),
        recall_score(y_test, y_pred_rf, average='weighted')
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_nb, average='weighted'),
        f1_score(y_test, y_pred_rf, average='weighted')
    ]
})

print(results_ml)


# ============================================
# 8. DL (LSTM + GRU)
# ============================================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])

seq = tokenizer.texts_to_sequences(df['clean_text'])
X_pad = pad_sequences(seq, maxlen=100)

le = LabelEncoder()
y_enc = le.fit_transform(df['domain'])

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_pad, y_enc)


# ============================================
# LSTM
# ============================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model_lstm = Sequential([
    Embedding(10000, 128),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(len(set(y_enc)), activation='softmax')
])

model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X_train_dl, y_train_dl, epochs=5)

# ============================================
# GRU
# ============================================
from tensorflow.keras.layers import GRU

model_gru = Sequential([
    Embedding(10000, 128),
    GRU(128),
    Dense(64, activation='relu'),
    Dense(len(set(y_enc)), activation='softmax')
])

model_gru.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gru.fit(X_train_dl, y_train_dl, epochs=5)


# ============================================
# DL METRICS
# ============================================
import numpy as np

y_pred_lstm = np.argmax(model_lstm.predict(X_test_dl), axis=1)
y_pred_gru = np.argmax(model_gru.predict(X_test_dl), axis=1)

results_dl = pd.DataFrame({
    "Model": ["LSTM", "GRU"],
    "Accuracy": [
        accuracy_score(y_test_dl, y_pred_lstm),
        accuracy_score(y_test_dl, y_pred_gru)
    ],
    "Precision": [
        precision_score(y_test_dl, y_pred_lstm, average='weighted'),
        precision_score(y_test_dl, y_pred_gru, average='weighted')
    ],
    "Recall": [
        recall_score(y_test_dl, y_pred_lstm, average='weighted'),
        recall_score(y_test_dl, y_pred_gru, average='weighted')
    ],
    "F1 Score": [
        f1_score(y_test_dl, y_pred_lstm, average='weighted'),
        f1_score(y_test_dl, y_pred_gru, average='weighted')
    ]
})

print(results_dl)


# ============================================
# 9. TRANSFORMERS
# ============================================
!pip install transformers datasets -q

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

labels = df['domain'].unique().tolist()
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

df['label'] = df['domain'].map(label2id)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'].tolist(),
    df['label'].tolist()
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_enc = tokenizer(train_texts, truncation=True, padding=True)
test_enc = tokenizer(test_texts, truncation=True, padding=True)

train_ds = Dataset.from_dict({
    'input_ids': train_enc['input_ids'],
    'attention_mask': train_enc['attention_mask'],
    'labels': train_labels
})

test_ds = Dataset.from_dict({
    'input_ids': test_enc['input_ids'],
    'attention_mask': test_enc['attention_mask'],
    'labels': test_labels
})

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()
bert_results = trainer.evaluate()


# ============================================
# 10. FINAL COMPARISON
# ============================================
results_transformer = pd.DataFrame({
    "Model": ["BERT"],
    "Accuracy": [bert_results['eval_accuracy']]
})

all_results = pd.concat([results_ml, results_dl, results_transformer])
print(all_results)
