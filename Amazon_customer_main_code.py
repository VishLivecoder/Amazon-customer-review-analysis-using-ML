# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:14:03 2021

@author: vishw
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix



#let us first import the dataset for some Exploratory data analysis.
data= pd.read_csv('reviews.csv')
print(data["Text"])
print(data["Summary"])

#cleaning the dataset.
#let us remove all the reviews that may not be useful for our problem at hand.
#
reviews=data[data['Score'] != 3]
reviews.shape

#sorting the reviews based on the product ID

sorted_reviews=reviews.sort_values(by = 'ProductId',axis=0,ascending=True,inplace=False, kind='quicksort',na_position='last')
sorted_reviews.shape

#removing the duplicate reviews.

deduplicated_review=sorted_reviews.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})
deduplicated_review.shape

#sorting the reviews by time. 
#this is because , we want to make sure that the model is tested on the recent reviews rather than old reviews.

sorted_reviews=deduplicated_review.sort_values(by = 'Time',axis=0,ascending=True,inplace=False, kind='quicksort',na_position='last')

#now let us convert the scores given for the product by the user into 0 and 1 

reviews_scoreset=sorted_reviews['Score']
def partition(var):
  if var>3:
    return 1
  else:
    return 0

sorted_reviews['Score']=reviews_scoreset.map(partition)

final_dataset=sorted_reviews

final_dataset.shape

  
final_interest=final_dataset["Summary"]+"."+final_dataset["Text"]

final_interest.head(5)


x=final_interest
y=final_dataset["Score"]
print(x.head(5))
final_interest.shape
x.shape
y.shape


##let us split the data into training and testing sets

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,stratify=y)
#stratify is used to have equal distribution of both classes [0 and 1] in both train and test data


X_train,X_cv,Y_train,Y_cv=train_test_split(X_train,Y_train,test_size=0.33,stratify=Y_train)
print("Training data shape ="+str(X_train.shape))

#BOW
#Vectorizing the Text features using BOW
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer=CountVectorizer(min_df=10,max_features=10000)
Vectorizer.fit(X_train.values.astype('U'))

X_train_essay_BOW=Vectorizer.transform(X_train.values.astype('U'))
X_test_essay_BOW=Vectorizer.transform(X_test.values.astype('U'))
X_cv_essay_BOW=Vectorizer.transform(X_cv.values.astype('U'))

list_of_words_essay_BOW=Vectorizer.get_feature_names()


#Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

Vectorizer=TfidfVectorizer(min_df=10,max_features=10000)

Vectorizer.fit(X_train.values.astype('U'))

X_train_essay_tfidf=Vectorizer.transform(X_train.values.astype('U'))
X_test_essay_tfidf=Vectorizer.transform(X_test.values.astype('U'))
X_cv_essay_tfidf=Vectorizer.transform(X_cv.values.astype('U'))

list_of_words_essay_tfidf=Vectorizer.get_feature_names()

print("After Bag Of Words Vectorization, the shape of Essay features are follows")
print("Train_Essay"+str(X_train_essay_BOW.shape))
print("Test_Essay"+str(X_test_essay_BOW.shape))
print("CV_Essay"+str(X_cv_essay_BOW.shape))
print("\n ")


print("After tfidf Vectorization, the shape of Essay features are follows")
print("Train_Essay"+str(X_train_essay_tfidf.shape))
print("Test_Essay"+str(X_test_essay_tfidf.shape))
print("CV_Essay"+str(X_cv_essay_tfidf.shape))

#now our input feature that is the combination of Summary and Review are converted into vectors

X_train_using_BOW_Normalizer=X_train_essay_BOW
X_test_using_BOW_Normalizer=X_test_essay_BOW
X_cv_using_BOW_Normalizer=X_cv_essay_BOW

X_train_using_tfidf_Normalizer=X_train_essay_tfidf
X_test_using_tfidf_Normalizer=X_test_essay_tfidf
X_cv_using_tfidf_Normalizer=X_cv_essay_tfidf



print("Final Data matrix after BOW Vectorization and OHE of categorical features and Normalization of Numerical features")
print(X_train_using_BOW_Normalizer.shape, Y_train.shape)
print(X_test_using_BOW_Normalizer.shape, Y_test.shape)
print(X_cv_using_BOW_Normalizer.shape, Y_cv.shape)
print("="*100)

print("Final Data matrix after Tfidf Vectorization and OHE of categorical features and Normalization of Numerical features")
print(X_train_using_tfidf_Normalizer.shape, Y_train.shape)
print(X_test_using_tfidf_Normalizer.shape, Y_test.shape)
print(X_cv_using_tfidf_Normalizer.shape, Y_cv.shape)




#Applying NaiveBayes on BOW Vectorization and plotting the AUC_ROC curve

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm


Alpha=[0.00001,0.0001,0.001,0.01,0.1,1.10,100,1000,10000]


train_auc = []
cv_auc = []

for i in tqdm(Alpha):
  Classifier=MultinomialNB(alpha=i)
  Classifier.fit(X_train_using_BOW_Normalizer,Y_train)

  
  y_train_predict=Classifier.predict_log_proba((X_train_using_BOW_Normalizer))[:,1]
  y_cv_predict=Classifier.predict_log_proba((X_cv_using_BOW_Normalizer))[:,1]

  train_auc.append(roc_auc_score(Y_train,y_train_predict))
  cv_auc.append(roc_auc_score(Y_cv,y_cv_predict))

plt.plot(Alpha,train_auc,label="Train AUC")
plt.plot(Alpha,cv_auc,label="CV AUC")

plt.scatter(Alpha,train_auc,label='Train AUC points')
plt.scatter(Alpha,cv_auc,label='CV AUC points')


plt.legend()
plt.xlabel("Alpha : Hyperparameter")
plt.ylabel("AUC")
plt.title("Error Plots for BOW")
plt.grid()
plt.show()


#Applying NaiveBayes on Tfidf Vectorization and plotting the AUC_ROC curve



Alpha=[0.00001,0.0001,0.001,0.01,0.1,1.10,100,1000,10000]


train_auc = []
cv_auc = []

for i in tqdm(Alpha):
  Classifier=MultinomialNB(alpha=i)
  Classifier.fit(X_train_using_BOW_Normalizer,Y_train)

  
  y_train_predict=Classifier.predict_log_proba((X_train_using_tfidf_Normalizer))[:,1]
  y_cv_predict=Classifier.predict_log_proba((X_cv_using_tfidf_Normalizer))[:,1]

  train_auc.append(roc_auc_score(Y_train,y_train_predict))
  cv_auc.append(roc_auc_score(Y_cv,y_cv_predict))

plt.plot(Alpha,train_auc,label="Train AUC")
plt.plot(Alpha,cv_auc,label="CV AUC")

plt.scatter(Alpha,train_auc,label='Train AUC points')
plt.scatter(Alpha,cv_auc,label='CV AUC points')


plt.legend()
plt.xlabel("Alpha : Hyperparameter")
plt.ylabel("AUC")
plt.title("Error Plots For tfidf")
plt.grid()
plt.show()

 #From the above graph we can estiimate the best alpha to be 74
from sklearn.metrics import roc_curve, auc
import numpy as np


print( "Let us now Predict the probablity scores for BOW vectorization" )

best_alpha=1

ClassifierBOW=MultinomialNB(alpha=best_alpha)
ClassifierBOW.fit(X_train_using_BOW_Normalizer,Y_train)
y_train_predict=ClassifierBOW.predict_log_proba((X_train_using_BOW_Normalizer))[:,1]
y_test_predict=ClassifierBOW.predict_log_proba((X_test_using_BOW_Normalizer))[:,1]


train_fpr,train_tpr,train_threshold=roc_curve(Y_train,y_train_predict)
test_fpr,test_tpr,test_threshold=roc_curve(Y_test,y_test_predict)


plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
test_AUC_BOW=auc(test_fpr, test_tpr)
plt.legend()
plt.xlabel("Alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS USING BOW")
plt.grid()
plt.show()


print( "Let us now Predict the probablity scores for Tfidf vectorization" )



Classifiertfidf=MultinomialNB(alpha=best_alpha)
Classifiertfidf.fit(X_train_using_tfidf_Normalizer,Y_train)
y_train_predict=Classifiertfidf.predict_log_proba((X_train_using_tfidf_Normalizer))[:,1]
y_test_predict=Classifiertfidf.predict_log_proba((X_test_using_tfidf_Normalizer))[:,1]


train_fpr,train_tpr,train_threshold=roc_curve(Y_train,y_train_predict)
test_fpr,test_tpr,test_threshold=roc_curve(Y_test,y_test_predict)


plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
test_AUC_TFIDF=auc(test_fpr, test_tpr)
plt.legend()
plt.xlabel("Alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS USING Tfidf")
plt.grid()
plt.show()


print("="*100)
from sklearn.metrics import confusion_matrix
def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    
    return t


best_t = find_best_threshold(train_threshold, train_fpr, train_tpr)

cf1=(confusion_matrix(Y_train, predict_with_best_t(y_train_predict, best_t)))
cf2=(confusion_matrix(Y_test, predict_with_best_t(y_test_predict, best_t)))

import seaborn as sns
print("Train confusion matrix")
labels = ['TrueNeg','False Pos','False Neg','True Pos']
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf1.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf1.flatten()/np.sum(cf1)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf1, annot=labels, fmt='', cmap='Blues')


print("Test Confusion matrix")
labels = ['TrueNeg','False Pos','False Neg','True Pos']
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf2.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf2.flatten()/np.sum(cf2)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf2, annot=labels, fmt='', cmap='Blues')

##
##
##Applying KNN Algorithm on the vectorized input.
##
##

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

k=[1,3,5,9,10,30,50,70,100]
cvscores=[]
testscores=[]
X_train_using_tfidf_Normalizer=X_train_using_tfidf_Normalizer[0:]
X_cv_using_tfidf_Normalizer=X_cv_using_tfidf_Normalizer[0:]
Y_train=Y_train[0:]
Y_cv=Y_cv[0:]
                                                        
for i in tqdm(k):
    KNN=KNeighborsClassifier()
    KNN.n_neighbors=i
    KNN.fit(X_train_using_tfidf_Normalizer,Y_train)
    Y_predicted = KNN.predict(X_cv_using_tfidf_Normalizer)
    cvscores.append(accuracy_score(Y_cv, Y_predicted.round()))

print(cvscores)


KNN.n_neighbors=10
Y_predicted = KNN.predict(X_test_using_tfidf_Normalizer)
testscores.append(accuracy_score(Y_test, Y_predicted.round())) 

print(testscores)

print("Let us implement Decision Tree classifier")

from sklearn.tree import DecisionTreeClassifier
cv_scores=[]
testscore=0
depth=[5,10,15,20,25,30,35,40]
for i in tqdm(depth):
    
    classifier_model = DecisionTreeClassifier(max_depth=i)
    classifier_model.fit(X_train_using_tfidf_Normalizer,Y_train)

    Y_predicted = classifier_model.predict(X_cv_using_tfidf_Normalizer)

    acc=accuracy_score(Y_cv, Y_predicted.round())
    cv_scores.append(acc)
    
print(cv_scores)



from mlxtend.plotting import plot_confusion_matrix
def confusionMatrixCalculation(ypred,ytest):
    
    y_pred=np.asarray(ypred)
    y_test=np.asarray(ytest)
    
    #finding the True Positive
    t_p = np.sum(np.logical_and(y_pred == 1 , y_test == 1))
    
    #finding the True Negative
    t_n = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    
    #finding the false positive
    f_p = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    
    #finding the false negative
    f_n = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    
    binary1 = np.array([[t_p,f_n],[f_p,t_n]])
    
    fig, ax = plot_confusion_matrix(conf_mat=binary1,show_absolute=True,
                                show_normed=True,
                                colorbar=True)
    plt.show()
    
    
    
    
    
    return [[t_p,f_n],[f_p,t_n]]

print("let us see how the accuracy improves by using bagging")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier




DT= DecisionTreeClassifier(criterion="entropy",max_depth=20)
   
        
        
classifier= BaggingClassifier(base_estimator=DT,n_estimators=20)
classifier.fit(X_train_using_tfidf_Normalizer,Y_train)
y_pred=classifier.predict(X_cv_using_tfidf_Normalizer)

confusionMatrixCalculation(y_pred,Y_cv)





print("Using SVM with RBF kernel for String input")
    
from sklearn import svm
from sklearn.metrics import accuracy_score
c=[1,3,5,9,10,30,50,70,100]
cvscores=[]
testscores=[]
for i in tqdm(c):
        
    clf=svm.SVC(kernel='rbf',c=i)
    clf.fit(X_train_using_tfidf_Normalizer,Y_train)
    y_pred=clf.predict(X_cv_using_tfidf_Normalizer)
    acc=accuracy_score(Y_cv, Y_predicted.round())
    cv_scores.append(acc)
print(cv_scores)
max_value = max(cv_scores)
    
max_index = cv_scores.index(max_value)

clf=svm.SVC(kernel='rbf',C=c[max_index])
clf.fit(X_train_using_tfidf_Normalizer,Y_train)
y_pred=clf.predict(X_test_using_tfidf_Normalizer)
acc=accuracy_score(Y_cv, Y_predicted.round())
print(acc)
confusionMatrixCalculation(y_pred,Y_test)
cvscores=cv_scores[8:17]
print(cvscores)

plt.plot(c,cvscores,label="Accuracy Plots")
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("Accuracy")
plt.title("Accuracy PLOTS USING Tfidf")
plt.grid()
plt.show()





    
    
    
   




 



    
     
     
     
     







