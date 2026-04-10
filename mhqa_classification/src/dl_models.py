from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout


def build_lstm(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def build_gru(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        GRU(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model