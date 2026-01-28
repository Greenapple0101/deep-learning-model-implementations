"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")

def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    
    # Flatten the tensors
    x_train_flat = x_train.view(num_train, -1)
    x_test_flat = x_test.view(num_test, -1)
    
    # Initialize the distance matrix
    dists = x_train.new_zeros(num_train, num_test)
    
    # Compute the squared Euclidean distance using two loops
    for i in range(num_train):
        for j in range(num_test):
            dists[i, j] = torch.sum((x_train_flat[i] - x_test_flat[j]) ** 2)
    
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    
    # Flatten the tensors
    x_train_flat = x_train.view(num_train, -1)
    x_test_flat = x_test.view(num_test, -1)
    
    # Initialize the distance matrix
    dists = x_train.new_zeros(num_train, num_test)
    
    # Loop over the training examples and compute distances in a vectorized manner
    for i in range(num_train):
        dists[i] = torch.sum((x_train_flat[i] - x_test_flat) ** 2, dim=1)
    
    return dists

def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    
    # Flatten the tensors
    x_train_flat = x_train.view(num_train, -1)
    x_test_flat = x_test.view(num_test, -1)
    
    # Compute the squared Euclidean distance using a fully vectorized approach
    # Formula: (x - y)^2 = x^2 + y^2 - 2 * x * y
    x_train_squared = torch.sum(x_train_flat ** 2, dim=1).view(-1, 1)  # (num_train, 1)
    x_test_squared = torch.sum(x_test_flat ** 2, dim=1).view(1, -1)    # (1, num_test)
    
    dists = x_train_squared + x_test_squared - 2 * torch.mm(x_train_flat, x_test_flat.t())
    
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
  
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)

    for i in range(num_test):

        _, knn_indices = torch.topk(dists[:, i], k=k, largest=False)
        
        knn_labels = y_train[knn_indices]
        
       
        y_pred[i] = knn_labels.bincount().argmax()
    
    return y_pred


import torch
import numpy as np
import matplotlib.pyplot as plt


import torch

class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        # Store the training data and labels
        self.x_train = x_train.view(x_train.shape[0], -1)  # Flatten to (num_train, C*H*W)
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        # Flatten x_test to (num_test, C*H*W)
        x_test = x_test.view(x_test.shape[0], -1)

        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]

        # Compute the L2 distance between test points and training points
        dists = torch.cdist(x_test, self.x_train, p=2)  # Pairwise Euclidean distances

        # Find the indices of the k nearest neighbors for each test point
        knn_indices = torch.topk(dists, k=k, dim=1, largest=False).indices

        # Retrieve the labels of the k nearest neighbors
        knn_labels = self.y_train[knn_indices]

        # Predict the most common label (majority vote) for each test point
        y_test_pred = torch.mode(knn_labels, dim=1).values

        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        # Predict labels using the KNN classifier
        y_test_pred = self.predict(x_test, k=k)

        # Compute accuracy by comparing predicted labels with true labels
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples

        # Print the accuracy if quiet is False
        if not quiet:
            print(f"Got {num_correct} / {num_samples} correct; accuracy is {accuracy:.2f}%")

        return accuracy


import torch
from typing import List, Dict
from collections import defaultdict

def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    # Use torch.chunk to divide the data into num_folds folds
    x_train_folds = torch.chunk(x_train, num_folds)
    y_train_folds = torch.chunk(y_train, num_folds)

    # Dictionary to hold accuracies for each value of k
    k_to_accuracies = defaultdict(list)

    # Perform cross-validation for each value of k
    for k in k_choices:
        # Perform cross-validation by training on num_folds-1 folds and validating on the last fold
        for fold in range(num_folds):
            # Prepare validation fold
            x_val_fold = x_train_folds[fold]
            y_val_fold = y_train_folds[fold]

            # Use the remaining folds as the training data
            x_train_fold = torch.cat([x_train_folds[i] for i in range(num_folds) if i != fold], dim=0)
            y_train_fold = torch.cat([y_train_folds[i] for i in range(num_folds) if i != fold], dim=0)

            # Initialize the classifier with the current training data
            classifier = KnnClassifier(x_train_fold, y_train_fold)

            # Predict the labels on the validation fold using the current k
            y_val_pred = classifier.predict(x_val_fold, k=k)

            # Calculate the accuracy and store it
            accuracy = (y_val_pred == y_val_fold).float().mean().item()
            k_to_accuracies[k].append(accuracy)

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = None
    best_mean_accuracy = -1

    # Check if k_to_accuracies is empty
    if not k_to_accuracies:
        print("k_to_accuracies is empty! Returning default k=1.")
        return 1  # Default value for k

    # Iterate over each k and its list of accuracies
    for k, accuracies in k_to_accuracies.items():
        # Calculate the mean accuracy for this k
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f"k={k}, mean_accuracy={mean_accuracy}")  # Debugging

        # Select the best k: if the mean accuracy is higher or if there is a tie, choose the smaller k
        if mean_accuracy > best_mean_accuracy or (mean_accuracy == best_mean_accuracy and (best_k is None or k < best_k)):
            best_k = k
            best_mean_accuracy = mean_accuracy

    print(f"Selected best k={best_k}, with mean accuracy={best_mean_accuracy}")  # Debugging
    return best_k

