"""
Supervised Classification Models for Resume Screening (Selected: Yes/No)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV


def get_classification_models():
    """
    Returns a dictionary of all required classification models:
    - Logistic Regression (interpretable baseline)
    - K-Nearest Neighbors (KNN)
    - Support Vector Machines (SVM): Linear and RBF kernel variations
    - Naive Bayes: Gaussian, Multinomial, Bernoulli
    - Tree-based: Decision Trees (Gini and Entropy criteria), Random Forest
    - Neural Networks / ANN: Multi-layer Perceptron (Perceptron learning)
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'SVM (Linear Kernel)': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            method='sigmoid', cv=3
        ),
        'SVM (RBF Kernel)': SVC(
            kernel='rbf', C=1.0, gamma='scale', max_iter=2500, random_state=42
        ),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Naive Bayes (Bernoulli)': BernoulliNB(binarize=0.0),
        'Naive Bayes (Multinomial)': MultinomialNB(),
        'Decision Tree (Gini)': DecisionTreeClassifier(
            criterion='gini', max_depth=12, min_samples_split=10, random_state=42
        ),
        'Decision Tree (Entropy)': DecisionTreeClassifier(
            criterion='entropy', max_depth=12, min_samples_split=10, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_split=8, 
            random_state=42, n_jobs=-1
        ),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=250, early_stopping=True, random_state=42
        )
    }
    return models
