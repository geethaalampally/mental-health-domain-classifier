from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tfidf_features(texts):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(texts).toarray()
    return X, tfidf


def dl_tokenizer(texts, vocab_size=10000, max_len=100):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    seq = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    return padded, tokenizer