from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt

def plot_precision_treshold(y_true, y_proba):
    precision, recall, thresholds =precision_recall_curve(y_true, y_proba)
    plt.plot(thresholds, precision[:-1])
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs Threshold')
    plt.grid(True)
    plt.show()
    
    return precision, recall, thresholds