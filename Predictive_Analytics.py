# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
#from sklearn import linear_model
from sklearn import metrics 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#import scikitplot as skplt

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    Accuracy=(true_positives+true_negatives)/total examples
    """
    tp_tn=sum(y_true==y_pred)
    total=np.prod(y_pred.shape)
    return tp_tn/total


def Recall(y_true,y_pred):
    

    """:type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    recall=(true_poitives/truepositives+falsenegatives)
    recall=correctly predicted/actual """

    recall=[]
    y_pred_iter=np.unique(y_pred)
    for ele in np.nditer(y_pred_iter): 
        recall.append(sum((y_true==ele) & (y_pred==ele))/sum(y_true==ele))
    return sum(recall)/len(recall)    
    
def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    precision=(true positives/true positives+false posities)
    precision=(correctly predicted/all predicted)
    """
    precision=[]
    y_pred_iter=np.unique(y_pred)
    for ele in np.nditer(y_pred_iter):
        precision.append(sum((y_true==ele) & (y_pred==ele))/sum(y_pred==ele))
    return sum(precision)/len(precision)    
    
    
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    n=np.prod(np.unique(y_pred).shape)
    c=np.zeros((n,n))
    y_i=np.unique(y_pred)
    y_i=np.transpose(y_i.reshape(-1,1))
    y_i=np.repeat(y_i,len(y_i[0]),axis=0)
    for i,row in enumerate(y_i):
        for j,col in enumerate(row):
            c[i][j]=sum((y_true==row[i]) & (y_pred==col))
    return c        

def KNN(X_train,X_test,Y_train,K):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :type K :int
    :rtype: numpy.ndarray
    """
    knN=[]
    count=0
    for x in X_test:
        print('in knn',count)
        count +=1
        e_distance=[]
        for y in X_train:
           e_distance.append(np.sqrt(np.sum((x-y)**2)))
        index_k=np.argsort(e_distance)[:K]
        k_near=[]
        for i in index_k:
            k_near.append(Y_train[i])
        knN.append(max(k_near,key=k_near.count))
    knN=np.array(knN)
    return knN
       
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
""" def DecisionTree(X_train,Y_train,X_test):
     gini_list=[]
     count=np.bincount(Y_train)
     p_y=count/len(Y_train)
     for p in p_y:
         gini_list.append(p**2
     gini=1-np.sum(gini_list)                
     
 """    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    clusters=[[] for _ in range(N)]

    centroids=[]
    row,col=X_train.shape
    old_centroid=np.zeros((N,col))
    i_rand= np.random.randint(0,row,size=N)
    for i in i_rand:
        centroids.append(X_train[i])
    
    d=1
    count=0
    while(d!=0 ): 
        count +=1
         
        for i,row in enumerate(X_train):
            e_distance=[]
            for j,col in enumerate(centroids):
                e_distance.append(np.sqrt(np.sum((row-col)**2)))
            nearest_points=np.argmin(e_distance)
            clusters[nearest_points].append(row)
            
        for i,data in enumerate(centroids):
            old_centroid[i]=data
            centroids[i]=np.mean(clusters[i],axis=0)
        for i in range(N):
            distance=[]
            distance.append((np.sqrt(np.sum((old_centroid[i]-centroids[i])**2))))
            d=sum(distance)
            print('distance after each loop',d)
        if count ==10:
            print('exit due to loop count')
            break
    print(centroids)    
    return clusters

def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    """df = pd.read_csv('C:\Masters\DataIntensiveComputing\Assignment1\data_1.csv')
    df[:] = np.nan_to_num(df)
    y = df[df.columns[-1]]
    X = df.drop(df.columns[-1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)"""
    svc = svm.SVC(kernel='linear')
    svc.fit(X_train, Y_train)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    dTree = tree.DecisionTreeRegressor()
    dTree.fit(X_train, Y_train)
    kNN = KNeighborsClassifier(n_neighbors=12)
    kNN.fit(X_train, Y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_kNN = kNN.predict(X_test)
    y_pred_dTree=dTree.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    #y_pred = [y_pred_lr,y_pred_kNN,y_pred_dTree,y_pred_svc] 
    y_pred = y_pred_lr
    return y_pred 


def SklearnVotingClassifier(X_train,Y_train,X_test):
    svc = svm.SVC()
    svc.fit(X_train, Y_train)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    dTree = tree.DecisionTreeRegressor()
    dTree.fit(X_train, Y_train)
    kNN = KNeighborsClassifier()
    kNN.fit(X_train, Y_train)
    est_ensemble = [('LogReg',lr),('SVC',svc),('DTree',dTree)]
    clf_voting = VotingClassifier(est_ensemble,voting='hard')
    clf_voting.fit(X_train, Y_train)
    y_pred = clf_voting.predict(X_test)
    print("accuracy is ",metrics.accuracy_score(y_pred,))
    return y_pred

df = pd.read_csv('C:\\Users\\maruthi pawan\\Desktop\\DIC\\Assignment1\\data.csv')
df[:] = np.nan_to_num(df)
y = df[df.columns[-1]]
X = df.drop(df.columns[-1], axis=1)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01)
y_pred_votingClassifier=SklearnVotingClassifier(X_train,y_train,X_test)
print("accuracy of voting classifier is ",metrics.accuracy_score(y_pred_votingClassifier,y_test))
#sc=StandardScaler()
#sc=MinMaxScaler(feature_range=(0, 1), copy=True)
#X_train=sc.fit_transform(X_train)
#X_test=sc.transform(X_test)
#clf=SklearnSupervisedLearning(X_train,y_train,X_test)
#y_train=y_train.to_numpy()
#clf2=KNN(X_train,X_test,y_train,12)
#clf=KNeighborsClassifier(n_neighbors=12)
#clf.fit(X_train,y_train)
#y_pred=clf.predict(X_test)
#print(y_pred-clf2)
#print(y_pred)
#print(clf2)
#print((clf2==y_pred).all())
#clf2=KMeans(n_clusters=8, init='random',n_jobs=-2).fit(X_train)
#clf=Kmeans(X_train,8)
#print(y_test)
#print(clf2.cluster_centers_)
#print(y_test.shape)   
#result=metrics.accuracy_score(clf,y_test)
#result=metrics.accuracy_score(clf2,y_test)
#cm=metrics.confusion_matrix(y_test,clf)
#cm1=ConfusionMatrix(y_test,clf)
#prec=metrics.average_precision_score(clf,y_test)
#res1=Accuracy(y_test,clf)
#prec1=Precision(y_test,clf)
#recall1=Recall(y_test,clf)
#print('accuracy  prec using scipy',result)
#print('accuracy prec without scipy',res1,prec1,recall1)


def SklearnGridSearch(X_train,Y_train,X_test):
    svc = svm.SVC(kernel='linear')
    svc.fit(X_train, Y_train)
    dTree = tree.DecisionTreeRegressor()
    dTree.fit(X_train, Y_train)
    kNN = KNeighborsClassifier()
    kNN.fit(X_train, Y_train)
    parameters=[{'C':[1,10,50,100],'kernel':['linear','rbf','poly']}]
    parameters_dTree=[{'criterion':['mse','friedman_mse'],'random_state':[10,100],'splitter':['best','random']}]
    parameters_kNN=[{'n_neighbors':[10,15],'p':[1,2],'weights':['uniform','distance']}]
    grid_search_svm=GridSearchCV(estimator=svc,param_grid=parameters,scoring='accuracy',n_jobs=-1,cv=5)
    grid_search=grid_search_svm.fit(X_train,Y_train)
    accuracy=grid_search.best_score_
    print(grid_search.best_params_)
    return accuracy

#gs=SklearnGridSearch(X_train,y_train,X_test)
#print(gs)
def ConfusionMatrix_Visualisation(X_train,Y_train,X_test,Y_test):
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    y_pred = lr.predict(X_test)
    
    cm=ConfusionMatrix(Y_test,y_pred)
    cm1=confusion_matrix(Y_test,y_pred)
    fig,ax=plt.subplots(2,3,1)
    
    
    
ConfusionMatrix_Visualisation(X_train,y_train,X_test,y_test)    
    