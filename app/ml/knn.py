"""
K-Nearest Neighbors implementation from scratch.
Supports cosine and Euclidean distance with uniform and distance weighting.
"""
import math
from typing import List, Tuple, Literal
from ..ml.features import SparseMatrix, SparseVector


class KNNClassifier:
    """K-Nearest Neighbors classifier implemented from scratch."""
    
    def __init__(self):
        self.X_train: SparseMatrix = None
        self.y_train: List[int] = None
        self.is_fitted = False
    
    def fit(self, X: SparseMatrix, y: List[int]) -> 'KNNClassifier':
        """Fit the KNN model by storing training data."""
        self.X_train = X
        self.y_train = y.copy()
        self.is_fitted = True
        return self
    
    def _cosine_distance(self, v1: SparseVector, v2: SparseVector) -> float:
        """Compute cosine distance between two normalized vectors."""
        # For L2-normalized vectors, cosine distance = 1 - dot_product
        dot_product = v1.dot(v2)
        # Clamp to avoid numerical issues
        dot_product = max(-1.0, min(1.0, dot_product))
        return 1.0 - dot_product
    
    def _euclidean_distance(self, v1: SparseVector, v2: SparseVector) -> float:
        """Compute Euclidean distance between two sparse vectors."""
        # For sparse vectors: ||v1 - v2||^2 = ||v1||^2 + ||v2||^2 - 2*v1·v2
        norm1_sq = sum(val * val for val in v1.values)
        norm2_sq = sum(val * val for val in v2.values)
        dot_product = v1.dot(v2)
        
        distance_sq = norm1_sq + norm2_sq - 2 * dot_product
        # Avoid numerical issues
        distance_sq = max(0.0, distance_sq)
        return math.sqrt(distance_sq)
    
    def _get_k_nearest(
        self, 
        query: SparseVector, 
        k: int, 
        metric: Literal["cosine", "euclidean"]
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors and their distances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Compute distances to all training points
        distances = []
        for i, train_vector in enumerate(self.X_train.vectors):
            if metric == "cosine":
                dist = self._cosine_distance(query, train_vector)
            else:  # euclidean
                dist = self._euclidean_distance(query, train_vector)
            distances.append((i, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def predict(
        self, 
        X: SparseMatrix, 
        k: int = 5, 
        metric: Literal["cosine", "euclidean"] = "cosine",
        weighting: Literal["uniform", "distance"] = "uniform"
    ) -> List[int]:
        """Predict classes for query points."""
        predictions = []
        
        for query_vector in X.vectors:
            # Get k nearest neighbors
            neighbors = self._get_k_nearest(query_vector, k, metric)
            
            # Count votes with optional distance weighting
            class_votes = {}
            total_weight = 0.0
            
            for neighbor_idx, distance in neighbors:
                neighbor_class = self.y_train[neighbor_idx]
                
                if weighting == "uniform":
                    weight = 1.0
                else:  # distance weighting
                    # Use inverse distance, with small epsilon to avoid division by zero
                    weight = 1.0 / (distance + 1e-8)
                
                if neighbor_class not in class_votes:
                    class_votes[neighbor_class] = 0.0
                class_votes[neighbor_class] += weight
                total_weight += weight
            
            # Find class with highest weighted vote
            if class_votes:
                predicted_class = max(class_votes.keys(), key=lambda c: class_votes[c])
            else:
                # Fallback (shouldn't happen with k > 0)
                predicted_class = 0
            
            predictions.append(predicted_class)
        
        return predictions
    
    def predict_proba(
        self, 
        X: SparseMatrix, 
        k: int = 5, 
        metric: Literal["cosine", "euclidean"] = "cosine",
        weighting: Literal["uniform", "distance"] = "uniform"
    ) -> List[List[float]]:
        """Predict class probabilities for query points."""
        probabilities = []
        
        # Get unique classes from training data
        unique_classes = sorted(set(self.y_train))
        n_classes = len(unique_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        for query_vector in X.vectors:
            # Get k nearest neighbors
            neighbors = self._get_k_nearest(query_vector, k, metric)
            
            # Count votes with optional distance weighting
            class_votes = [0.0] * n_classes
            total_weight = 0.0
            
            for neighbor_idx, distance in neighbors:
                neighbor_class = self.y_train[neighbor_idx]
                class_idx = class_to_idx[neighbor_class]
                
                if weighting == "uniform":
                    weight = 1.0
                else:  # distance weighting
                    weight = 1.0 / (distance + 1e-8)
                
                class_votes[class_idx] += weight
                total_weight += weight
            
            # Normalize to get probabilities
            if total_weight > 0:
                proba = [vote / total_weight for vote in class_votes]
            else:
                # Uniform distribution fallback
                proba = [1.0 / n_classes] * n_classes
            
            probabilities.append(proba)
        
        return probabilities
    
    def get_neighbors_for_explanation(
        self, 
        query: SparseVector, 
        k: int = 5, 
        metric: Literal["cosine", "euclidean"] = "cosine"
    ) -> List[Tuple[int, float, int]]:
        """Get nearest neighbors with their indices, distances, and labels for explanation."""
        neighbors = self._get_k_nearest(query, k, metric)
        return [(idx, dist, self.y_train[idx]) for idx, dist in neighbors]


def cross_validate_knn(
    X: SparseMatrix, 
    y: List[int], 
    k_values: List[int],
    metrics: List[str],
    weightings: List[str],
    cv_folds: int = 5,
    seed: int = 42
) -> List[dict]:
    """Cross-validate KNN with different hyperparameters."""
    import random
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Create stratified folds (simple version)
    indices = list(range(len(y)))
    random.shuffle(indices)
    fold_size = len(indices) // cv_folds
    folds = []
    
    for i in range(cv_folds):
        start_idx = i * fold_size
        if i == cv_folds - 1:  # Last fold gets remaining samples
            end_idx = len(indices)
        else:
            end_idx = (i + 1) * fold_size
        folds.append(indices[start_idx:end_idx])
    
    results = []
    
    # Try all hyperparameter combinations
    for k in k_values:
        for metric in metrics:
            for weighting in weightings:
                fold_scores = []
                
                for fold_idx in range(cv_folds):
                    # Split data
                    test_indices = set(folds[fold_idx])
                    train_indices = [i for i in indices if i not in test_indices]
                    
                    # Create train/test splits
                    X_train_vectors = [X[i] for i in train_indices]
                    X_test_vectors = [X[i] for i in test_indices]
                    y_train_fold = [y[i] for i in train_indices]
                    y_test_fold = [y[i] for i in test_indices]
                    
                    X_train_fold = SparseMatrix(X_train_vectors)
                    X_test_fold = SparseMatrix(X_test_vectors)
                    
                    # Train and predict
                    knn = KNNClassifier()
                    knn.fit(X_train_fold, y_train_fold)
                    
                    y_pred = knn.predict(X_test_fold, k=k, metric=metric, weighting=weighting)
                    
                    # Compute accuracy
                    correct = sum(1 for true, pred in zip(y_test_fold, y_pred) if true == pred)
                    accuracy = correct / len(y_test_fold) if y_test_fold else 0.0
                    fold_scores.append(accuracy)
                
                # Average across folds
                mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
                std_score = 0.0
                if len(fold_scores) > 1:
                    variance = sum((score - mean_score) ** 2 for score in fold_scores) / len(fold_scores)
                    std_score = math.sqrt(variance)
                
                results.append({
                    'k': k,
                    'metric': metric,
                    'weighting': weighting,
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score,
                    'fold_scores': fold_scores
                })
    
    # Sort by mean CV score
    results.sort(key=lambda x: x['mean_cv_score'], reverse=True)
    return results