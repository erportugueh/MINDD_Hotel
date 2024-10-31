import pandas as pd
import numpy as np
import scipy as sp
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
    

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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

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

        return mean_scores
    
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
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=Y)
    
        scaler = StandardScaler()
        scaler.fit(X_train)
    
        # fit a Logistic Regression model and feature selection altogether 
        # select the Lasso (l1) penalty.
        # The selectFromModel class from sklearn, selects the features which coefficients are non-zero
    
        sel_ = SelectFromModel(LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))
    
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



# K-Means
def initialize_centroids(X, k):
    """Randomly initialize centroids"""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """Assign clusters based on closest centroid"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroids by computing the mean of all points assigned to each cluster"""
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if points.size:
            centroids[i] = points.mean(axis=0)
    return centroids

def kmeans(X, k, max_iters=100, tol=1e-4):
    """K-Means clustering algorithm"""
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return labels, centroids



#----------------------------------------------------------------------------------

# Naive Bayes
def NaiveBayes(self, X, Y):
    

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Apply K-Means
    k = 4
    labels, centroids = kmeans(X, k)
    
    #falta analisar o codigo e dar print dos dados  
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42, stratify=Y)

    # Initialize the Gaussian Naive Bayes classifier
    gnb = GaussianNB()

    # Train the classifier
    gnb.fit(X_train, y_train)

    # Make predictions
    y_pred = gnb.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


#----------------------------------------------------------------------------------
#Afonso


def majority_voting_classifiers(X, Y, classifiers, cv=5):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

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
def weighted_majority_voting_classifiers(X, Y, classifiers, test_size=0.2):
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
def stacking_logistic_regression(X, Y, base_classifiers, test_size=0.2):
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
def stacking_svc(X, Y, base_classifiers, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Train base classifiers and get their predictions
    base_predictions = np.zeros((X_train.shape[0], len(base_classifiers)))
    for i, (name, clf) in enumerate(base_classifiers):
        clf.fit(X_train, y_train)
        base_predictions[:, i] = clf.predict(X_train)

    # Train the meta-classifier (SVM) on the predictions of base classifiers
    meta_clf = SVC()
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
def bagging_classifier(X, Y, base_classifier, n_estimators=10, test_size=0.2):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Create a BaggingClassifier with the provided base classifier
    bagging_clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=n_estimators, random_state=42, stratify=y)

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


    
