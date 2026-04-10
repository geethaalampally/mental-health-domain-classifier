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


def preprocess_dataframe(df):
    df = df[['question', 'topic']]
    df.rename(columns={'topic': 'domain'}, inplace=True)
    df.dropna(subset=['question', 'domain'], inplace=True)

    df['clean_text'] = df['question'].apply(clean_text)
    return df