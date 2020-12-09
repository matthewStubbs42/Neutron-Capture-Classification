# Analysis, plotting
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

##......................................................................................................##
#                                         Confusion Matrix                                               #
##......................................................................................................##

def plot_confusion_matrix(dump_path, classes, model, fc, dw, k, normalize=True, cmap=plt.cm.Blues):
    
    y_true = np.genfromtxt(dump_path + 'y_true.csv', delimiter=",")
    y_pred = np.genfromtxt(dump_path + 'y_pred.csv', delimiter=",")
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize = (8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    title = model
    
    if fc:
        title += ", fully connected"
    else:
        title += ", knn graph, k=" + str(k) 
    if dw:
        title += ", edge weighted"
    plt.title(title)
    plt.savefig(dump_path + "confusion matrix.png", bbox_inches = 'tight')
    return


##......................................................................................................##
#                                     learning progression plot                                          #
##......................................................................................................##

def disp_learn_hist_smoothed(location, window_train, window_val, model, fc, dw, k, losslim=None, show=False):
    train_log=location+'/log_train.csv'
    val_log=location+'/log_val.csv'
    best_val_log=location +'/log_best_val.csv'
    
    train_log_csv = pd.read_csv(train_log)
    val_log_csv  = pd.read_csv(val_log)
    best_val_csv = pd.read_csv(best_val_log)

    epoch_train  = moving_average(np.array(train_log_csv.epoch),window_train)
    accuracy_train = moving_average(np.array(train_log_csv.accuracy),window_train)
    loss_train  = moving_average(np.array(train_log_csv.loss),window_train)
    
    epoch_val    = moving_average(np.array(val_log_csv.epoch),window_val)
    accuracy_val = moving_average(np.array(val_log_csv.accuracy),window_val)
    loss_val     = moving_average(np.array(val_log_csv.loss),window_val)

    fig, ax1 = plt.subplots(figsize=(16,12),facecolor='w')
    line11 = ax1.plot(epoch_train, loss_train, linewidth=2, label='Average training loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch_val, loss_val, label='Average validation loss', color='blue')
    
    
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    if losslim is not None:
        ax1.set_ylim(0.,losslim)
    
    ax2 = ax1.twinx()
    line21 = ax2.plot(epoch_train, accuracy_train, linewidth=2, label='Average training accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch_val, accuracy_val, label='Average validation accuracy', color='red')
        
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)
    
    ax3 = ax1.twinx()    
    ax3.scatter(best_val_csv.epoch, best_val_csv.accuracy, color = 'indigo', ls = '--', label='BEST validation accuracy')
    ax3.set_ylim(0.,1.0)
    ax3.tick_params('y',colors='r',labelsize=18)
    ax3.legend(loc='upper left')
    
    lines  = line11+ line12 + line21+ line22 
    
    labels = [l.get_label() for l in lines]
    leg = ax2.legend(lines, labels, fontsize=16, loc='best', numpoints=1)

    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')
    
    title = model
    if fc:
        title += ", fully connected"
    else:
        title += ", knn graph, k=" + str(k) 
    if dw:
        title += ", edge weighted"
        
    plt.title(title)
    plt.savefig(location + "training.png", bbox_inches = 'tight')

    if show:
        plt.grid()
        plt.show()
        return
    
    return fig

##......................................................................................................##
#                                          moving average                                                #
##......................................................................................................##

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


##......................................................................................................##
#                                           ROC plot                                                     #
##......................................................................................................##

def ROC(location, model, fc, dw, k, title=None):
    # Compute ROC curve and ROC area for each class
    
    true_label = pd.read_csv(location + '/y_true.csv')
    n_test_pred = pd.read_csv(location + '/nPred.csv')
    e_test_pred = pd.read_csv(location + '/ePred.csv')
    
    fpr0, tpr0, thresholds0 = roc_curve(true_label, n_test_pred)
    roc_auc0 = auc(fpr0, tpr0)

    fpr1, tpr1, thresholds1 = roc_curve(true_label, e_test_pred)
    roc_auc1 = auc(fpr1, tpr1)

    title = model
    if fc:
        title += ", fully connected"
    else:
        title += ", knn graph, k=" + str(k) 
    if dw:
        title += ", edge weighted"

    plt.figure(figsize = (12, 8))
    plt.plot(fpr1, tpr1, color='darkorange', lw=1, label='ncapture ROC_AUC: {:.3f}'.format(roc_auc1))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(location + "ROC_curve.png", bbox_inches = 'tight')

    
