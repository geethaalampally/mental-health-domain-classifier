def predict_text(text, model, vectorizer):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]