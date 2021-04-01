from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance

def display_roc_curve(model, inputs_test, outputs_test):
    predictions = model.predict(inputs_test)

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(outputs_test, predictions)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def display_permutation_importance(model, inputs_test, outputs_test, feature_names):
    perm = PermutationImportance(model, random_state=1).fit(inputs_test, outputs_test)
    return eli5.show_weights(perm, feature_names = feature_names)

def display_predictions(model, inputs_test, outputs_test):
    predictions = model.predict(inputs_test)

    print(predictions)
    print(np.max(predictions))
    print(np.sum(outputs_test))