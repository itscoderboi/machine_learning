from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
def training():
		l=[]
		mnist=fetch_mldata('MNIST ORIGINAL')
		X_train,X_test,y_train,y_test=train_test_split(mnist.data,mnist.target,test_size=10000,random_state=10)	
		std=StandardScaler()
		l.append(std)
		X_train=std.fit_transform(X_train.astype(np.float64))
		X_test=std.transform(X_test.astype(np.float64))
		pca=PCA(.95)
		l.append(pca)
		X_train_pca=pca.fit_transform(X_train)
		X_test_pca=pca.transform(X_test)
		log=LogisticRegression()
		l.append(log)
		log.fit(X_train_pca,y_train)
		rf=RandomForestClassifier()
		l.append(rf)
		rf.fit(X_train_pca,y_train)
		gb=GradientBoostingClassifier()
		l.append(gb)
		gb.fit(X_train_pca,y_train)
		f=open('training.txt','wb')
		pickle.dump(l,f)
training()