import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


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
def NaiveBayes():
    

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Apply K-Means
    k = 4
    labels, centroids = kmeans(X, k)
    
    #falta analisar o codigo e dar print dos dados  
    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

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



    
