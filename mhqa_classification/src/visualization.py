import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# ================================
# 1. CONFUSION MATRIX
# ================================
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ================================
# 2. MODEL ACCURACY BAR GRAPH
# ================================
def plot_model_accuracy(results_df):
    plt.figure(figsize=(8,5))
    plt.bar(results_df['Model'], results_df['Accuracy'])
    plt.xticks(rotation=30)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


# ================================
# 3. MULTI-METRIC COMPARISON
# ================================
def plot_all_metrics(results_df):
    x = np.arange(len(results_df))
    width = 0.2

    plt.figure(figsize=(12,6))

    plt.bar(x - width*1.5, results_df['Accuracy'], width, label='Accuracy')
    plt.bar(x - width/2, results_df['Precision'], width, label='Precision')
    plt.bar(x + width/2, results_df['Recall'], width, label='Recall')
    plt.bar(x + width*1.5, results_df['F1 Score'], width, label='F1 Score')

    plt.xticks(x, results_df['Model'], rotation=30)
    plt.title("All Models Comparison")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.legend()

    plt.show()


# ================================
# 4. ROC CURVE (FOR ML MODELS)
# ================================
def plot_roc_curve(models, X_test, y_test, class_labels):
    y_test_bin = label_binarize(y_test, classes=class_labels)

    plt.figure()

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# ================================
# 5. DL TRAINING CURVES
# ================================
def plot_training_history(history_lstm, history_gru):
    # Accuracy
    plt.figure()
    plt.plot(history_lstm.history['accuracy'], label='LSTM Train')
    plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val')

    plt.plot(history_gru.history['accuracy'], label='GRU Train')
    plt.plot(history_gru.history['val_accuracy'], label='GRU Val')

    plt.title("DL Models Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(history_lstm.history['loss'], label='LSTM Loss')
    plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')

    plt.plot(history_gru.history['loss'], label='GRU Loss')
    plt.plot(history_gru.history['val_loss'], label='GRU Val Loss')

    plt.title("DL Models Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# ================================
# 6. SAVE PLOTS (OPTIONAL BONUS)
# ================================
def save_plot(fig, path):
    fig.savefig(path, bbox_inches='tight')