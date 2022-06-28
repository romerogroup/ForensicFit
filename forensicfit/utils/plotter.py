import numpy as np
import matplotlib.pyplot as plt
import itertools


plt.rcParams["font.family"] = "Arial"
plt.rc("font", size=11)  # controls default text sizes
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels
plt.rc("legend", fontsize=12)  # legend fontsize


def plot_confusion_matrix(matrix: np.asarray,
                          class_names: np.asarray,
                          title: str='Confusion matrix',
                          cmap: str = 'Blues',
                          normalize: bool=False):
    

    accuracy = np.trace(matrix) / float(np.sum(matrix))
    misclass = 1 - accuracy

    cmap = plt.get_cmap(cmap)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)#, rotation=45)
        plt.yticks(tick_marks, class_names)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


    thresh = matrix.max() / 1.5 if normalize else matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.show()