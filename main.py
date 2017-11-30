# coding=latin-1
import os
from pandas import DataFrame
import numpy
import tarfile
import time
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords

COMPLAINT = 'KLAGE'
NORMAL = 'ANDET'
poslabel=NORMAL
n_folds=1


def build_data_frame(path):
    rows = []
    index = []
    f = open(path)
    irow=0
    for row in f.readlines():
        row=row.replace(";)","blinkesmiley")
        row=row.replace(";-)","blinkesmiley")
        split=row.decode('latin-1').strip().split(';')
        if irow>0:
            try:
                rows.append({'text': split[0], 'class': split[1]})
                index.append(irow)
            except IndexError,e:
                test = 0
        irow+=1

    data_frame = DataFrame(rows, index=index)
    return data_frame


def runWithoutFolding():
    global predictions
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    k_fold = KFold(n=len(data), n_folds=2)
    train_indices = range(0,int(len(data)*5./10.))
    test_indices = range(int(len(data)*5./10.),len(data)-1)
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values

    # print 'fitting w cross validation,k_fold: '
    pipeline.fit(train_text, train_y)
    # print 'predicting w cross validation,k_fold: '
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=poslabel)
    scores.append(score)
# print('Total documents classified:', len(data))
    print('Score:', sum(scores) / len(scores))  # ,' Correct Thematic reco:',confusion[1])
    print('Confusion matrix:')
    print(confusion / float(len(data)))

def folding(n_folds=6):
    global predictions
    k_fold = KFold(n=len(data), n_folds=n_folds)
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values

        #print 'fitting w cross validation,k_fold: '
        pipeline.fit(train_text, train_y)
        #print 'predicting w cross validation,k_fold: '
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label=poslabel)
        scores.append(score)
    #print('Total documents classified:', len(data))
    print('Score:', sum(scores) / len(scores))#,' Correct Thematic reco:',confusion[1])
    print('Confusion matrix:')
    print(confusion/float(len(data)))

print 'importing data'
data = DataFrame({'text': [], 'class': []})
path = 'data/klager_simpel.csv'
####### OBS ######## Facebook post har tit ; i smileys - tjek lige om data bliver indlæst korrekt.
#data = data_frame=DataFrame.from_csv(path, header=0, sep=';', index_col=None, encoding='latin-1')
data = data.append(build_data_frame(path))

print 'reindiexing'
data = data.reindex(numpy.random.permutation(data.index))

print 'trying with SVM'
# try with support vector machine as in http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
for ngram in range(2,3):
    #The 'log' loss gives logistic regression, a probabilistic classifier.
        # 'modified_huber' is another smooth loss that brings tolerance to
        # outliers as well as probability estimates.
        # 'squared_hinge' is like hinge but is quadratically penalized.
        # 'perceptron' is the linear loss used by the perceptron algorithm.
        # The other losses are designed for regression but can be useful in
        # classification as well; see SGDRegressor for a description.
    for loss in ['log','hinge','modified_huber']:
        pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(ngram_range=(1, ngram),stop_words=stopwords.words('danish'))),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', SGDClassifier(loss=loss, penalty='l2', alpha=1e-3, random_state=42, n_jobs=-1))])
        t0 = time.time()
        if n_folds==1:
            runWithoutFolding()
        else:
            folding(n_folds=n_folds)
        t1 = time.time()
        print 'Time to calc with 1 to'+str(ngram)+'-gram and td-idf transformation and support vector machine classifier: with loss: '+loss + str(t1 - t0)

        print ' '

print 'Trying with ngram and tf-idf and multinomialNB (baysian propb, assuming independence equal weights)'
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 3),stop_words=stopwords.words('danish'))),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

t0 = time.time()
if n_folds == 1:
    runWithoutFolding()
else:
    folding(n_folds=n_folds)
t1 = time.time()
print 'Time to calc with bigram and tf-idf: ' + str(t1 - t0)

print ' '
print 'Trying with SVM'
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 3),stop_words=stopwords.words('danish'))),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', LinearSVC(penalty='l2', loss='squared_hinge', tol=0.0001,random_state=42))])
t0 = time.time()
if n_folds == 1:
    runWithoutFolding()
else:
    folding(n_folds=n_folds)
t1 = time.time()
print 'Time to calc with bigram w. td-idf and linear support vector  classifier: ' + str(t1 - t0)

print 'Printing tfidf matrix'

results = []
for line in data['text']:
    results.extend(line.strip().split('\n'))

vectorizer= TfidfVectorizer(min_df=1)

X_train_tf=vectorizer.fit_transform(results)
print(X_train_tf.shape)
idf=vectorizer._tfidf.idf_

#p= (vectorizer.get_feature_names(), idf)
p = zip(vectorizer.get_feature_names(), idf)
p.sort(key = lambda t: t[1])
print p
#with open("tfidf.txt","w") as t:
#    for x in p:
#        print>>t, x