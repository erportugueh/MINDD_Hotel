# Essentials
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from lightgbm import LGBMClassifier

# Machine Learning and Model Selection
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, LeaveOneOut
)
from sklearn.utils import resample
from sklearn.feature_selection import SelectFromModel

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from xgboost import XGBClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Set warnings to ignore
warnings.filterwarnings('ignore')


class Algoritmos:
    def determine_best_K(self, X, Y):
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        accuracies = []
        k_range=range(1, 21)
        # Testing different values of k
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Plotting the accuracy for each value of k
        plt.figure(figsize=(10, 5))
        plt.plot(k_range, accuracies, marker='o')
        plt.title('K-Value vs Accuracy')
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Accuracy')
        plt.xticks(k_range)
        plt.grid()
        plt.show()

        # Finding the best k
        best_k = k_range[accuracies.index(max(accuracies))]
        print(f"The best k value is: {best_k} with an accuracy of: {max(accuracies):.4f}")

        return best_k
    



    def knn_function(self, X, Y, k):
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        # Creating the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fitting the model
        knn.fit(X_train, y_train)

        # Making predictions
        y_pred = knn.predict(X_test)
        acc=accuracy_score(y_test, y_pred)
        # Evaluating the model
        print("Accuracy:", acc*100)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix", confusion_matrix(y_test, y_pred))
        return acc, knn


    def crossValidation_knn(self, X, Y, cv, k):
        # Scale the entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation with KNN and get predictions
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, X_scaled, Y, cv=cv)

        # Calculate mean accuracy
        mean_scores = np.mean(cross_val_score(knn, X_scaled, Y, cv=cv, scoring='accuracy')) * 100
        print(f"KNN model accuracy with {cv}-fold cross-validation (in %):", mean_scores)

        # Generate and print classification report
        print("\nClassification Report:\n", classification_report(Y, y_pred))
        print(f"Confusion Matrix\n", confusion_matrix(Y, y_pred))
        return mean_scores, knn
    def loo_knn(self, X, Y, k):

        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with
        loo = LeaveOneOut()
        knn = KNeighborsClassifier(n_neighbors=k)

        scores = cross_val_score(knn, X_scaled, Y, cv=loo, scoring='accuracy')
        Y_pred= cross_val_predict(knn, X_scaled, Y, cv=loo)


        mean_scores = np.mean(scores) * 100
        print(f"K nearest Neighbors model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification report:\n", classification_report(Y, Y_pred))
        print(f"Confusion Matrix: \n", confusion_matrix(Y, Y_pred))

        return mean_scores, knn
    
    def bootstrap_knn(self, X_train, Y_train, X_test, Y_test, n, k):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize K-Nearest Neighbors model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, Y_train)
        y_pred = knn.predict(X_test)

        # Perform bootstrapping with KNN
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            knn.fit(X_bs, y_bs)
            score = knn.score(X_test_scaled, Y_test)
            scores.append(score)
            
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n", classification_report(Y_test, y_pred))
        print(f"Bootstrap Mean Accuracy:\n", confusion_matrix(Y_test, y_pred))
    
        return mean_score, knn
    

    def oversample_undersample_knn(self, X, Y, k):

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Separate the minority and majority classes
        X_train_minority = X_train[y_train == 1]
        y_train_minority = y_train[y_train == 1]
        X_train_majority = X_train[y_train == 0]
        y_train_majority = y_train[y_train == 0]

        # Oversample the minority class
        X_train_minority_oversampled, y_train_minority_oversampled = resample(
            X_train_minority, y_train_minority, 
            replace=True, # sample with replacement
            n_samples=len(X_train_majority), # to match the majority class
            random_state=42
        )

        # Combine oversampled minority class with majority class
        X_train_balanced = np.vstack((X_train_majority, X_train_minority_oversampled))
        y_train_balanced = np.hstack((y_train_majority, y_train_minority_oversampled))

        # Train the KNN classifier on oversampled data
        knn_oversampled = KNeighborsClassifier(n_neighbors=k)
        knn_oversampled.fit(X_train_balanced, y_train_balanced)
        y_pred_oversampled = knn_oversampled.predict(X_test)



        acc_oversampled=accuracy_score(y_test, y_pred_oversampled)

        print("Accuracy:", acc_oversampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_oversampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_oversampled))


        # Undersample the majority class
        X_train_majority_undersampled, y_train_majority_undersampled = resample(
            X_train_majority, y_train_majority, 
            replace=False, # sample without replacement
            n_samples=len(X_train_minority), # to match the minority class
            random_state=42
        )

        # Combine undersampled majority class with minority class
        X_train_balanced = np.vstack((X_train_majority_undersampled, X_train_minority))
        y_train_balanced = np.hstack((y_train_majority_undersampled, y_train_minority))

        # Train the KNN classifier on undersampled data
        knn_undersampled = KNeighborsClassifier(n_neighbors=5)
        knn_undersampled.fit(X_train_balanced, y_train_balanced)
        y_pred_undersampled = knn_undersampled.predict(X_test)


        acc_undersampled=accuracy_score(y_test, y_pred_undersampled)

        print("Accuracy:", acc_undersampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_undersampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_undersampled))
        return acc_oversampled, acc_undersampled, knn_oversampled, knn_undersampled
    

  
    #----------Naive_bayes--------
    def NB_function(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        gnb= GaussianNB()
        gnb.fit(X_train, y_train)
        Y_pred=gnb.predict(X_test)
        acc = accuracy_score (y_test, Y_pred)
        print("Accuracy:", acc*100)
        print("\nClassification Report:\n", classification_report(y_test, Y_pred))
        print("\nConfusion Matrix\n", confusion_matrix(y_test, Y_pred))

        return acc*100, gnb
    
    def crossValidation_NB(self, X, Y, cv):
        # Scale the entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation 
        gnb = GaussianNB()
        y_pred = cross_val_predict(gnb, X_scaled, Y, cv=cv)

        # Calculate mean accuracy
        mean_scores = np.mean(cross_val_score(gnb, X_scaled, Y, cv=cv, scoring='accuracy')) * 100
        print(f"KNN model accuracy with {cv}-fold cross-validation (in %):", mean_scores)

        # Generate and print classification report
        print("\nClassification Report:\n", classification_report(Y, y_pred))
        print(f"Confusion Matrix\n", confusion_matrix(Y, y_pred))

        return mean_scores, gnb
        
    def loo_NB(self, X, Y):
        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with
        loo = LeaveOneOut()
        gnb = GaussianNB()

        scores = cross_val_score(gnb, X_scaled, Y, cv=loo, scoring='accuracy')
        Y_pred= cross_val_predict(gnb, X_scaled, Y, cv=loo)


        mean_scores = np.mean(scores) * 100
        print(f"K nearest Neighbors model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification report:\n", classification_report(Y, Y_pred))
        print(f"Confusion Matrix\n", confusion_matrix(Y, Y_pred))
        return mean_scores, gnb
    
    def bootstrap_NB(self, X_train, Y_train, X_test, Y_test, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize K-Nearest Neighbors model
        gnb = GaussianNB()
        gnb.fit(X_train_scaled, Y_train)
        y_pred = gnb.predict(X_test)

        # Perform bootstrapping 
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            gnb.fit(X_bs, y_bs)
            score = gnb.score(X_test_scaled, Y_test)
            scores.append(score)
            
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n", classification_report(Y_test, y_pred))
        print(f"Bootstrap Mean Accuracy:\n", confusion_matrix(Y_test, y_pred))
    
        return mean_score, gnb
    
    def oversample_undersample_NB(self, X, Y):
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Separate the minority and majority classes
        X_train_minority = X_train[y_train == 1]
        y_train_minority = y_train[y_train == 1]
        X_train_majority = X_train[y_train == 0]
        y_train_majority = y_train[y_train == 0]

        # Oversample the minority class
        X_train_minority_oversampled, y_train_minority_oversampled = resample(
            X_train_minority, y_train_minority, 
            replace=True, # sample with replacement
            n_samples=len(X_train_majority), # to match the majority class
            random_state=42
        )

        # Combine oversampled minority class with majority class
        X_train_balanced = np.vstack((X_train_majority, X_train_minority_oversampled))
        y_train_balanced = np.hstack((y_train_majority, y_train_minority_oversampled))

        # Train the oversampled data
        gnb_oversampled = KNeighborsClassifier()
        gnb_oversampled.fit(X_train_balanced, y_train_balanced)
        y_pred_oversampled = gnb_oversampled.predict(X_test)



        acc_oversampled=accuracy_score(y_test, y_pred_oversampled)

        print("Accuracy:", acc_oversampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_oversampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_oversampled))


        # Undersample the majority class
        X_train_majority_undersampled, y_train_majority_undersampled = resample(
            X_train_majority, y_train_majority, 
            replace=False, # sample without replacement
            n_samples=len(X_train_minority), # to match the minority class
            random_state=42
        )

        # Combine undersampled majority class with minority class
        X_train_balanced = np.vstack((X_train_majority_undersampled, X_train_minority))
        y_train_balanced = np.hstack((y_train_majority_undersampled, y_train_minority))

        # Train theclassifier on undersampled data
        gnb_undersampled = KNeighborsClassifier(n_neighbors=5)
        gnb_undersampled.fit(X_train_balanced, y_train_balanced)
        y_pred_undersampled = gnb_undersampled.predict(X_test)


        acc_undersampled=accuracy_score(y_test, y_pred_undersampled)

        print("Accuracy:", acc_undersampled*100)

        print("Classification Report (Undersampled):\n", classification_report(y_test, y_pred_undersampled))
        print("Confusion Matrix (Undersampled):\n", confusion_matrix(y_test, y_pred_undersampled))
        return acc_oversampled, acc_undersampled, gnb_oversampled, gnb_undersampled
      
    
    #----------SVM---------

    def svc_function(self, X, Y):
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        # Creating the svc classifier
        scv = SVC(kernel='linear', max_iter=1000)

        # Fitting the model
        scv.fit(X_train, y_train)

        # Making predictions
        y_pred = scv.predict(X_test)
        acc=accuracy_score(y_test, y_pred)
        # Evaluating the model
        print("Accuracy:", acc*100)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix", confusion_matrix(y_test, y_pred))
        return acc, scv


    def crossValidation_svc(self, X, Y, cv):
        # Scale the entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation with scv and get predictions
        scv = SVC(kernel='linear', max_iter=1000)
        y_pred = cross_val_predict(scv, X_scaled, Y, cv=cv)

        # Calculate mean accuracy
        mean_scores = np.mean(cross_val_score(scv, X_scaled, Y, cv=cv, scoring='accuracy')) * 100
        print(f"KNN model accuracy with {cv}-fold cross-validation (in %):", mean_scores)

        # Generate and print classification report
        print("\nClassification Report:\n", classification_report(Y, y_pred))
        print(f"Confusion Matrix\n", confusion_matrix(Y, y_pred))
        return mean_scores, scv
    
    def loo_scv(self, X, Y):

        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with
        loo = LeaveOneOut()
        scv = SVC(kernel='linear', max_iter=1000)

        scores = cross_val_score(scv, X_scaled, Y, cv=loo, scoring='accuracy')
        Y_pred= cross_val_predict(scv, X_scaled, Y, cv=loo)


        mean_scores = np.mean(scores) * 100
        print(f"K nearest Neighbors model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification report:\n", classification_report(Y, Y_pred))
        print(f"Confusion Matrix: \n", confusion_matrix(Y, Y_pred))

        return mean_scores, scv
    
    def bootstrap_scv(self, X_train, Y_train, X_test, Y_test, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize scv model
        scv = SVC(kernel='linear', max_iter=1000)
        scv.fit(X_train_scaled, Y_train)
        y_pred = scv.predict(X_test)

        # Perform bootstrapping with scv
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            scv.fit(X_bs, y_bs)
            score = scv.score(X_test_scaled, Y_test)
            scores.append(score)
            
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n", classification_report(Y_test, y_pred))
        print(f"Bootstrap Mean Accuracy:\n", confusion_matrix(Y_test, y_pred))
    
        return mean_score, scv
    

    def oversample_undersample_scv(self, X, Y):

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Separate the minority and majority classes
        X_train_minority = X_train[y_train == 1]
        y_train_minority = y_train[y_train == 1]
        X_train_majority = X_train[y_train == 0]
        y_train_majority = y_train[y_train == 0]

        # Oversample the minority class
        X_train_minority_oversampled, y_train_minority_oversampled = resample(
            X_train_minority, y_train_minority, 
            replace=True, # sample with replacement
            n_samples=len(X_train_majority), # to match the majority class
            random_state=42
        )

        # Combine oversampled minority class with majority class
        X_train_balanced = np.vstack((X_train_majority, X_train_minority_oversampled))
        y_train_balanced = np.hstack((y_train_majority, y_train_minority_oversampled))

        # Train the scv classifier on oversampled data
        scv_oversampled = SVC(kernel='linear', max_iter=1000)
        scv_oversampled.fit(X_train_balanced, y_train_balanced)
        y_pred_oversampled = scv_oversampled.predict(X_test)



        acc_oversampled=accuracy_score(y_test, y_pred_oversampled)

        print("Accuracy:", acc_oversampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_oversampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_oversampled))


        # Undersample the majority class
        X_train_majority_undersampled, y_train_majority_undersampled = resample(
            X_train_majority, y_train_majority, 
            replace=False, # sample without replacement
            n_samples=len(X_train_minority), # to match the minority class
            random_state=42
        )

        # Combine undersampled majority class with minority class
        X_train_balanced = np.vstack((X_train_majority_undersampled, X_train_minority))
        y_train_balanced = np.hstack((y_train_majority_undersampled, y_train_minority))

        # Train the scv classifier on undersampled data
        scv_undersampled = SVC(kernel='linear', max_iter=1000)
        scv_undersampled.fit(X_train_balanced, y_train_balanced)
        y_pred_undersampled = scv_undersampled.predict(X_test)


        acc_undersampled=accuracy_score(y_test, y_pred_undersampled)

        print("Accuracy:", acc_undersampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_undersampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_undersampled))
        return acc_oversampled*100, acc_undersampled, scv_oversampled, scv_undersampled
    

      
    #-----------Decision Tree---------

    def dt_function(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        dt = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        acc=accuracy_score(y_test, y_pred)

        print("Accuracy:", acc*100)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix", confusion_matrix(y_test, y_pred))
        return acc*100, dt


    def crossValidation_dt(self, X, Y, cv):
        # Scale the entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation with scv and get predictions
        dt = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)
        y_pred = cross_val_predict(dt, X_scaled, Y, cv=cv)

        # Calculate mean accuracy
        mean_scores = np.mean(cross_val_score(dt, X_scaled, Y, cv=cv, scoring='accuracy')) * 100
        print(f"KNN model accuracy with {cv}-fold cross-validation (in %):", mean_scores)

        # Generate and print classification report
        print("\nClassification Report:\n", classification_report(Y, y_pred))
        print(f"Confusion Matrix\n", confusion_matrix(Y, y_pred))
        return mean_scores, dt
    
    def loo_dt(self, X, Y):

        # Scale entire feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Leave-One-Out cross-validation with
        loo = LeaveOneOut()
        dt = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)


        scores = cross_val_score(dt, X_scaled, Y, cv=loo, scoring='accuracy')
        Y_pred= cross_val_predict(dt, X_scaled, Y, cv=loo)


        mean_scores = np.mean(scores) * 100
        print(f"K nearest Neighbors model accuracy with leave-one-out cross-validation (in %):", mean_scores)
        print(f"Classification report:\n", classification_report(Y, Y_pred))
        print(f"Confusion Matrix: \n", confusion_matrix(Y, Y_pred))

        return mean_scores, dt
    
    def bootstrap_dt(self, X_train, Y_train, X_test, Y_test, n):
        scores = []
        
        # Scale train and test sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize dt model
        dt = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)

        dt.fit(X_train_scaled, Y_train)
        y_pred = dt.predict(X_test)

        # Perform bootstrapping with scv
        for _ in range(n):
            X_bs, y_bs = resample(X_train_scaled, Y_train)
            dt.fit(X_bs, y_bs)
            score = dt.score(X_test_scaled, Y_test)
            scores.append(score)
            
        mean_score = np.mean(scores) * 100
        print(f"Bootstrap Mean Accuracy: {mean_score:.2f}%")
        print(f"Classification Report:\n", classification_report(Y_test, y_pred))
        print(f"Bootstrap Mean Accuracy:\n", confusion_matrix(Y_test, y_pred))
    
        return mean_score, dt
    

    def oversample_undersample_dt(self, X, Y):

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Separate the minority and majority classes
        X_train_minority = X_train[y_train == 1]
        y_train_minority = y_train[y_train == 1]
        X_train_majority = X_train[y_train == 0]
        y_train_majority = y_train[y_train == 0]

        # Oversample the minority class
        X_train_minority_oversampled, y_train_minority_oversampled = resample(
            X_train_minority, y_train_minority, 
            replace=True, # sample with replacement
            n_samples=len(X_train_majority), # to match the majority class
            random_state=42
        )

        # Combine oversampled minority class with majority class
        X_train_balanced = np.vstack((X_train_majority, X_train_minority_oversampled))
        y_train_balanced = np.hstack((y_train_majority, y_train_minority_oversampled))

        # Train the scv classifier on oversampled data
        dt_oversampled = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)
        dt_oversampled.fit(X_train_balanced, y_train_balanced)
        y_pred_oversampled = dt_oversampled.predict(X_test)



        acc_oversampled=accuracy_score(y_test, y_pred_oversampled)

        print("Accuracy:", acc_oversampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_oversampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_oversampled))


        # Undersample the majority class
        X_train_majority_undersampled, y_train_majority_undersampled = resample(
            X_train_majority, y_train_majority, 
            replace=False, # sample without replacement
            n_samples=len(X_train_minority), # to match the minority class
            random_state=42
        )

        # Combine undersampled majority class with minority class
        X_train_balanced = np.vstack((X_train_majority_undersampled, X_train_minority))
        y_train_balanced = np.hstack((y_train_majority_undersampled, y_train_minority))

        # Train the scv classifier on undersampled data
        dt_undersampled = DecisionTreeClassifier(criterion='gini',max_depth=10, min_samples_split=50, min_samples_leaf=25, random_state=42)

        dt_undersampled.fit(X_train_balanced, y_train_balanced)
        y_pred_undersampled = dt_undersampled.predict(X_test)


        acc_undersampled=accuracy_score(y_test, y_pred_undersampled)

        print("Accuracy:", acc_undersampled*100)

        print("Classification Report (Oversampled):\n", classification_report(y_test, y_pred_undersampled))
        print("Confusion Matrix (Oversampled):\n", confusion_matrix(y_test, y_pred_undersampled))
        return acc_oversampled, acc_undersampled, dt_oversampled, dt_undersampled
    

    def FunctionChisq(self, inpData, TargetVariable, CategoricalVariablesList):
        from scipy.stats import chi2_contingency
        
        # Creating an empty list of final selected predictors
        SelectedPredictors=[]
    
        for predictor in CategoricalVariablesList:
            CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
            ChiSqResult = chi2_contingency(CrossTabResult)
            
            # If the ChiSq P-Value is <0.05, that means we reject H0
            if (ChiSqResult[1] < 0.05):
                print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
                SelectedPredictors.append(predictor)
            else:
                print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])        
    
        
        return(SelectedPredictors)

    def FunctionAnova(self, inpData, TargetVariable, ContinuousPredictorList):
        from scipy.stats import f_oneway

    
        # Creating an empty list of final selected predictors
        SelectedPredictors=[]
        
        print('##### ANOVA Results ##### \n')
        for predictor in ContinuousPredictorList:
            CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
            AnovaResults = f_oneway(*CategoryGroupLists)
            
            # If the ANOVA P-Value is <0.05, that means we reject H0
            if (AnovaResults[1] < 0.05):
                print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
                SelectedPredictors.append(predictor)
            else:
                print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
        
        return(SelectedPredictors)
        
    def lasso_regularization(self, X, Y):
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    
        scaler = StandardScaler()
        scaler.fit(X_train)
    
        # fit a Logistic Regression model and feature selection altogether 
        # select the Lasso (l1) penalty.
        # The selectFromModel class from sklearn, selects the features which coefficients are non-zero
    
        sel_ = SelectFromModel(LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=10))
    
        sel_.fit(scaler.transform(X_train), y_train)
    
        # make a list with the selected features
        selected_feat = X_train.columns[(sel_.get_support())]
        
        print("Number of features which coefficient was shrank to zero: ", np.sum(sel_.estimator_.coef_ == 0))
        # identify the removed features like this:
        removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
        print(removed_feats) 
    
        # transform data
        X_lasso = pd.DataFrame(sel_.transform(scaler.transform(X)), columns=selected_feat)
        
        return X_lasso

        # Define function to create the MLP network
    def nnetClassif(self, inputDim, optimizer, neurons):
            # Initialize the Sequential model
            model = Sequential()
            
            # Input layer + hidden layer
            model.add(Dense(units=neurons, 
                            input_dim=inputDim, 
                            kernel_initializer='uniform', 
                            activation='relu'))
            
            # Second hidden layer
            model.add(Dense(units=neurons, 
                            kernel_initializer='uniform', 
                            activation='relu'))
            
            # Output layer
            model.add(Dense(units=1, 
                            kernel_initializer='uniform', 
                            activation='sigmoid'))
            
            # Compile the model
            model.compile(optimizer=optimizer, 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
            
            return model        







#----------------------------------------------------------------------------------
#Afonso


def majority_voting_classifiers(X, Y, classifiers, cv=5):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    # Create a VotingClassifier with the provided classifiers
    voting_clf = VotingClassifier(estimators=classifiers, voting='hard')

    # Train the VotingClassifier
    voting_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = voting_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy


    # Scale the feature set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create a VotingClassifier with the provided classifiers
    voting_clf = VotingClassifier(estimators=classifiers, voting='hard')

    # Perform cross-validation and get mean accuracy
    mean_scores = np.mean(cross_val_score(voting_clf, X_scaled, Y, cv=cv, scoring='accuracy')) * 100
    print(f"Majority Voting Classifier accuracy with {cv}-fold cross-validation (in %):", mean_scores)

    return mean_scores
#----------------------------------------------------------------------------------
def weighted_majority_voting_classifiers(X, Y, classifiers, test_size=0.3):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create a VotingClassifier with the provided classifiers and weights
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')

    # Train the VotingClassifier
    voting_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = voting_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#----------------------------------------------------------------------------------
def stacking_logistic_regression(X, Y, base_classifiers, test_size=0.3):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Train base classifiers and get their predictions
    base_predictions = np.zeros((X_train.shape[0], len(base_classifiers)))
    for i, (name, clf) in enumerate(base_classifiers):
        clf.fit(X_train, y_train)
        base_predictions[:, i] = clf.predict(X_train)

    # Train the meta-classifier (Logistic Regression) on the predictions of base classifiers
    meta_clf = LogisticRegression()
    meta_clf.fit(base_predictions, y_train)

    # Get predictions from base classifiers on the test set
    base_test_predictions = np.zeros((X_test.shape[0], len(base_classifiers)))
    for i, (name, clf) in enumerate(base_classifiers):
        base_test_predictions[:, i] = clf.predict(X_test)

    # Make final predictions using the meta-classifier
    final_predictions = meta_clf.predict(base_test_predictions)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, final_predictions)
    report = classification_report(y_test, final_predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#----------------------------------------------------------------------------------
def stacking_svc(X, Y, base_classifiers, test_size=0.3):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Train base classifiers and get their predictions
    base_predictions = np.zeros((X_train.shape[0], len(base_classifiers)))
    for i, (name, clf) in enumerate(base_classifiers):
        clf.fit(X_train, y_train)
        base_predictions[:, i] = clf.predict(X_train)

    # Train the meta-classifier (SVM) on the predictions of base classifiers
    meta_clf = SVC(kernel='linear')
    meta_clf.fit(base_predictions, y_train)

    # Get predictions from base classifiers on the test set
    base_test_predictions = np.zeros((X_test.shape[0], len(base_classifiers)))
    for i, (name, clf) in enumerate(base_classifiers):
        base_test_predictions[:, i] = clf.predict(X_test)

    # Make final predictions using the meta-classifier
    final_predictions = meta_clf.predict(base_test_predictions)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, final_predictions)
    report = classification_report(y_test, final_predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#----------------------------------------------------------------------------------
def bagging_classifier(X, Y, base_classifier, n_estimators=10, test_size=0.3):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create a BaggingClassifier with the provided base classifier
    bagging_clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=n_estimators, random_state=42, stratify=Y)

    # Train the BaggingClassifier
    bagging_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = bagging_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#----------------------------------------------------------------------------------
def boosting_adaboosting(X, Y, base_classifier, n_estimators=50, test_size=0.3):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create an AdaBoostClassifier with the provided base classifier
    adaboost_clf = AdaBoostClassifier(estimator=base_classifier, n_estimators=n_estimators, random_state=42)

    # Train the AdaBoostClassifier
    adaboost_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = adaboost_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#--------------------------------------------------------------------------------
def boosting_gradient_boosting(X, Y, n_estimators=100, test_size=0.3):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create a GradientBoostingClassifier
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

    # Train the GradientBoostingClassifier
    gb_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = gb_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#--------------------------------------------------------------------------------
def boosting_xgboost(X, Y, n_estimators=100, test_size=0.3):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create an XGBClassifier
    xgb_clf = XGBClassifier(n_estimators=n_estimators, random_state=42)

    # Train the XGBClassifier
    xgb_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
#--------------------------------------------------------------------------------
def boosting_lightgbm(X, Y, n_estimators=100, test_size=0.3):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create a LGBMClassifier
    lgbm_clf = LGBMClassifier(n_estimators=n_estimators, random_state=42)

    # Train the LGBMClassifier
    lgbm_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = lgbm_clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy
    
