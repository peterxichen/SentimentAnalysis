import sklearn
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
import numpy
import random
import matplotlib.pyplot as plt

''' Multinomial '''

# class labels
y_train = numpy.genfromtxt('train_classes_5.txt', delimiter = '\n')
y_test = numpy.genfromtxt('test_classes_0.txt', delimiter = '\n')

# frequency bag of words
x_train = numpy.genfromtxt('train_bag_of_words_5.csv', delimiter=',')
x_test = numpy.genfromtxt('test_bag_of_words_0.csv', delimiter=',')

# fit Multinomial NB classifier
model = MultinomialNB().fit(x_train, y_train)
y_pred = model.predict(x_test)
print "Multinomial NB accuracy: ", metrics.accuracy_score(y_test, y_pred)
conf_mat = metrics.confusion_matrix(y_test, y_pred)
tp = conf_mat[0][0]
fp = conf_mat[0][1]
fn = conf_mat[1][0]
tn = conf_mat[1][1]
precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
specificity = float(tn)/(fp+tn)
print "Confusion Matrix:\n", conf_mat
print "Precision: ", precision
print "Recall: ", recall
print "Specificity: ", specificity
print "False Positive Rate: ", 1-specificity
print "F1 score: ", 2*precision*recall/(precision+recall)

y_probs = model.predict_proba(x_test)
y_p = map(lambda z: z[1], y_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()
n = len(y_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_p)
roc_auc = metrics.auc(fpr, tpr)

print "AUC: ", roc_auc


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


''' Bernoulli '''
# class labels
y_train = numpy.genfromtxt('train_classes_5.txt', delimiter = '\n')
y_test = numpy.genfromtxt('test_classes_0.txt', delimiter = '\n')

# binary bag of words
x_train = numpy.genfromtxt('train_bag_of_words_5_binary.csv', delimiter=',')
x_test = numpy.genfromtxt('test_bag_of_words_0_binary.csv', delimiter=',')

# fit Bernoulli Naive Bayes classifier
model = BernoulliNB().fit(x_train, y_train)
y_pred = model.predict(x_test)

print "\n"
print "Bernoulli NB accuracy: ", metrics.accuracy_score(y_test, y_pred)
conf_mat = metrics.confusion_matrix(y_test, y_pred)
tp = conf_mat[0][0]
fp = conf_mat[0][1]
fn = conf_mat[1][0]
tn = conf_mat[1][1]
precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
specificity = float(tn)/(fp+tn)
print "Confusion Matrix:\n", conf_mat
print "Precision: ", precision
print "Recall: ", recall
print "Specificity: ", specificity
print "False Positive Rate: ", 1-specificity
print "F1 score: ", 2*precision*recall/(precision+recall)
y_probs = model.predict_proba(x_test)
y_p = map(lambda z: z[1], y_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()
n = len(y_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_p)
roc_auc = metrics.auc(fpr, tpr)

print "AUC: ", roc_auc

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
