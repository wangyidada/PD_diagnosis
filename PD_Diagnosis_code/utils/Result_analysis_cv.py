import numpy as np
from MeDIT.Quantification import AUC_Confidence_Interval
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, recall_score, precision_score
import itertools
import os

def cal_auc(label, score):
    [AUC, CI, sorted_scores] = AUC_Confidence_Interval(label, score, CI_index=0.95)
    fpr, tpr, thresholds = roc_curve(label, score)
    return AUC, fpr, tpr, thresholds


def plot_roc(train_AUC, train_fpr, train_tpr,
             val_AUC, val_fpr, val_tpr,
             test_AUC, test_fpr, test_tpr):
    lw = 1.5
    plt.figure(figsize=(5, 5))
    plt.plot(train_fpr, train_tpr, color='green', lw=lw, label='Training ROC (AUC = %0.2f)' % train_AUC)
    plt.plot(val_fpr, val_tpr, color='blue', lw=lw, label='Validation ROC (AUC = %0.2f)' % val_AUC)
    plt.plot(test_fpr, test_tpr, color='red', lw=lw, label='Testing ROC (AUC = %0.2f)' % test_AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="lower right")
    plt.title('ROC')
    plt.savefig('ROC.png', dpi = 300)
    plt.show()


def evaluation(fpr, tpr, t, score, label, save_path=None):
    RightIndex = (tpr + (1 - fpr) - 1)
    index = np.where(RightIndex == np.max(RightIndex))[0]
    threshold = t[index]
    for z in range(len(index)):
        test_threshold = t[index[z]]
        score[np.where(score < threshold)] = 0
        score[np.where(score >= threshold)] = 1

        C = confusion_matrix(label, score)
        plot_confusion_matrix(C, save_path=save_path)

        TP = C[1, 1]
        TN = C[0, 0]
        FP = C[0, 1]
        FN = C[1, 0]
        Sensitivity = TP / (TP + FN)
        Specificiry = TN / (FP + TN)
        PPV = TP / (TP + FP)
        NPV = TN / (FN + TN)
        ACC = (TP+TN) / len(label)
        Precision =precision_score(label, score)
        F1_score = f1_score(label, score)
        Recall_score = recall_score(label, score)

        print('accuracy is %0.3f, Sensitivity is %0.3f, Specificiry is %0.3f, PPV is %0.3f, NPV is %0.3f,'
              'F1 score is %0.3f, Recall score is %0.3f, Precision  is %0.3f,threshold is %0.3f, '
              % (ACC, Sensitivity, Specificiry, PPV, NPV,F1_score, Recall_score, Precision,  test_threshold))


def classificaton_evaluation(values):
    # values = [[train_label, train_score], [val_label, val_score], [test_label, test_score]]
    train_score = np.asarray(values[0][1], dtype=np.float)
    train_label = np.asarray(values[0][0], dtype=np.int)
    val_score = np.asarray(values[1][1], dtype=np.float)
    val_label = np.asarray(values[1][0], dtype=np.int)
    test_score = np.asarray(values[2][1], dtype=np.float)
    test_label = np.asarray(values[2][0], dtype=np.int)

    [train_AUC, train_fpr, train_tpr, train_thresholds] = cal_auc(train_label, train_score)
    [val_AUC, val_fpr, val_tpr, val_thresholds] = cal_auc(val_label, val_score)
    [test_AUC, test_fpr, test_tpr, test_thresholds] = cal_auc(test_label, test_score)
    plot_roc(train_AUC, train_fpr, train_tpr,val_AUC, val_fpr, val_tpr,test_AUC, test_fpr, test_tpr)

    print('Trian Dataset result:\n')
    evaluation(train_fpr, train_tpr, train_thresholds, train_score, train_label)
    print('Val Dataset result: \n')
    evaluation(val_fpr, val_tpr, val_thresholds, val_score, val_label)
    print('Test Dataset result: \n')
    evaluation(test_fpr, test_tpr, test_thresholds, test_score, test_label)

def classificaton_single_evaluation(values, save_fig_file):
    # values = [[train_label, train_score]]
    train_score = np.asarray(values[0][1], dtype=np.float)
    train_label = np.asarray(values[0][0], dtype=np.int)
    [train_AUC, train_fpr, train_tpr, train_thresholds] = cal_auc(train_label, train_score)
    evaluation(train_fpr, train_tpr, train_thresholds, train_score, train_label, save_fig_file)
    plot_roc_curve([[train_label, train_score]], names=['Testing dataset'], save_name=os.path.join(save_fig_file,'roc.jpg' ))



def classificaton_cv_evaluation(values, save_fig_file):
    # values = [[train_label, train_score],[test_label, test_score]]
    train_score = np.asarray(values[0][1], dtype=np.float)
    train_label = np.asarray(values[0][0], dtype=np.int)
    test_score = np.asarray(values[1][1], dtype=np.float)
    test_label = np.asarray(values[1][0], dtype=np.int)

    [train_AUC, train_fpr, train_tpr, train_thresholds] = cal_auc(train_label, train_score)
    [test_AUC, test_fpr, test_tpr, test_thresholds] = cal_auc(test_label, test_score)
    plot_roc_curve([[train_label, train_score], [test_label, test_score]], names=['Training dataset', 'Testing dataset'], save_name=os.path.join(save_fig_file,'roc.jpg' ))

    print('Trian Dataset result:\n')
    evaluation(train_fpr, train_tpr, train_thresholds, train_score, train_label, save_fig_file)
    print('Test Dataset result: \n')
    evaluation(test_fpr, test_tpr, test_thresholds, test_score, test_label, save_fig_file)


def plot_roc_curve(values, colors=['blue', 'green', 'purple'], names=['Train', 'Val', 'Test'], save_name=None):
    # values = [[train_label, train_score], [val_label, val_score], [test_label, test_score]]
    for i in range(len(values)):
        labels = np.asarray(values[i][0], dtype=np.int)
        logits = np.asarray(values[i][1], dtype=np.float)
        FPR, TPR, t = roc_curve(labels, logits)
        roc_auc = auc(FPR, TPR)
        plt.plot(FPR, TPR, colors[i], label=names[i] +' (AUC = %0.2f)' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if save_name != None:
        plt.savefig(save_name, dpi=300)
    plt.show()


def plot_confusion_matrix(cm, classes=['NC', 'PD'], normalize=False, cmap=plt.cm.Blues, save_path=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 fontsize=30,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.tight_layout()

    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path!=None:
        plt.savefig(os.path.join(save_path,'confusion.png'), dpi=300)
    plt.show()



