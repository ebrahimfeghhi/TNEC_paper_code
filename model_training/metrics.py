import matplotlib
matplotlib.use('Agg')
import numpy as np 
from sklearn.metrics import matthews_corrcoef as mat
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import precision_recall_curve as prc 
from sklearn.metrics import roc_auc_score as auc
from matplotlib import pyplot as plt
import sklearn

# Input: prediction and ground truth labels
# Output: optimal threshold to convert probabilities to binary values, computed using ROC curve 
def create_ROC_plot(y_pred, y_test, results_storage_path):
    
    fpr, tpr, thresholds = roc(y_test, y_pred)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(results_storage_path + "/" + "ROC.png")
    plt.close()
    
    print("AUC: ", auc(y_test, y_pred))
    
    opt_thresh = thresholds[np.argmax(tpr-fpr)]
    return opt_thresh

# Input: prediction and ground truth labels
# Output: optimal threshold to convert probabilities to binary values, computed using Precision Recall Curve and fscore
def create_PRC_plot(y_pred, y_test, results_storage_path):
    precision, recall, thresholds = prc(y_test, y_pred)
    f_num = 2 * precision * recall
    f_dem = precision + recall 
    
    # prevent nan values when TP=0 
    fscore = np.divide(f_num, f_dem, out=np.zeros_like(f_num), where=f_dem!=0) 
   
    plt.title('Precision Recall Curve')
    plt.plot(recall, precision, 'b')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    if (results_storage_path != None):
        plt.savefig(results_storage_path + "/" + "PRC.png")
    plt.close()
    
    print("Max fscore: ", np.max(fscore))
    opt_thresh = thresholds[np.argmax(fscore)]
    return opt_thresh
    
    
def pred_to_int(y_pred, opt_thresh=None):
    
    if (opt_thresh == None):
        y_pred_int = np.argmax(y_pred, axis=1)
        return y_pred_int
    else:
        y_pred[y_pred > opt_thresh] = 1
        y_pred[y_pred != 1] = 0
        return y_pred

    
def metrics_calc(y_pred_int, y_test):
    matthew = mat(y_test, y_pred_int)
    classif_report = cr(y_test, y_pred_int, output_dict=True)
    confusion_mat = cm(y_test, y_pred_int)
    return  matthew, classif_report, confusion_mat




