"""
Support Vector Machine implementation from scratch using SMO algorithm.
Supports linear and RBF kernels for binary classification.
"""
import math
import random
from typing import List, Tuple, Literal, Optional, Dict, Any
from ..ml.features import SparseMatrix, SparseVector


class SVMClassifier:
    """SVM classifier implemented from scratch using SMO algorithm."""
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf"] = "linear",
        gamma: float = 0.1,
        tol: float = 1e-3,
        max_passes: int = 5,
        random_state: int = 42
    ):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state
        
        # Model parameters (learned during training)
        self.alphas: List[float] = []
        self.b: float = 0.0
        self.X_train: SparseMatrix = None
        self.y_train: List[int] = []
        self.support_vectors_: List[int] = []
        self.support_vector_alphas_: List[float] = []
        self.support_vector_labels_: List[int] = []
        self.is_fitted = False
        
        # Cache for kernel computations
        self._kernel_cache: Dict[Tuple[int, int], float] = {}
    
    def _kernel(self, i: int, j: int) -> float:
        """Compute kernel function between training samples i and j."""
        # Check cache first
        cache_key = (min(i, j), max(i, j))
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        xi = self.X_train[i]
        xj = self.X_train[j]
        
        if self.kernel_type == "linear":
            result = xi.dot(xj)
        elif self.kernel_type == "rbf":
            # RBF kernel: exp(-gamma * ||xi - xj||^2)
            # For sparse vectors: ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2*xi·xj
            norm_xi_sq = sum(v * v for v in xi.values)
            norm_xj_sq = sum(v * v for v in xj.values)
            dot_product = xi.dot(xj)
            
            distance_sq = norm_xi_sq + norm_xj_sq - 2 * dot_product
            distance_sq = max(0.0, distance_sq)  # Avoid numerical issues
            
            result = math.exp(-self.gamma * distance_sq)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Cache the result
        self._kernel_cache[cache_key] = result
        return result
    
    def _compute_error(self, i: int) -> float:
        """Compute prediction error for sample i."""
        # E_i = f(x_i) - y_i where f(x_i) = sum(alpha_j * y_j * K(x_i, x_j)) + b
        decision_value = 0.0
        for j in range(len(self.alphas)):
            if self.alphas[j] > 0:  # Only consider non-zero alphas
                decision_value += self.alphas[j] * self.y_train[j] * self._kernel(i, j)
        decision_value += self.b
        
        return decision_value - self.y_train[i]
    
    def _select_second_alpha(self, i1: int, E1: float) -> int:
        """Select second alpha using heuristic (maximum |E1 - E2|)."""
        max_delta = 0.0
        i2 = -1
        
        # Try to find alpha that maximizes |E1 - E2|
        for j in range(len(self.alphas)):
            if j == i1:
                continue
                
            E2 = self._compute_error(j)
            delta = abs(E1 - E2)
            
            if delta > max_delta:
                max_delta = delta
                i2 = j
        
        # If no good candidate found, select randomly
        if i2 == -1:
            candidates = [j for j in range(len(self.alphas)) if j != i1]
            if candidates:
                i2 = random.choice(candidates)
            else:
                i2 = 0  # Fallback
        
        return i2
    
    def _update_threshold(self) -> None:
        """Update the threshold b."""
        # Use support vectors to compute b
        support_alphas = []
        support_indices = []
        
        for i, alpha in enumerate(self.alphas):
            if 0 < alpha < self.C:  # Free support vectors
                support_alphas.append(alpha)
                support_indices.append(i)
        
        if not support_indices:
            # No free support vectors, use all support vectors
            support_indices = [i for i, alpha in enumerate(self.alphas) if alpha > 0]
        
        if support_indices:
            b_values = []
            for i in support_indices:
                decision_value = 0.0
                for j in range(len(self.alphas)):
                    if self.alphas[j] > 0:
                        decision_value += self.alphas[j] * self.y_train[j] * self._kernel(i, j)
                
                b_i = self.y_train[i] - decision_value
                b_values.append(b_i)
            
            # Average b values
            self.b = sum(b_values) / len(b_values)
        else:
            self.b = 0.0
    
    def _smo_step(self, i1: int, i2: int) -> bool:
        """Perform one SMO optimization step."""
        if i1 == i2:
            return False
        
        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1, y2 = self.y_train[i1], self.y_train[i2]
        
        # Compute bounds
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return False
        
        # Compute kernel values
        k11 = self._kernel(i1, i1)
        k12 = self._kernel(i1, i2)
        k22 = self._kernel(i2, i2)
        eta = k11 + k22 - 2 * k12
        
        if eta <= 0:
            return False  # Skip this pair
        
        # Compute errors
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        
        # Compute new alpha2
        alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
        
        # Clip alpha2
        if alpha2_new >= H:
            alpha2_new = H
        elif alpha2_new <= L:
            alpha2_new = L
        
        # Check if change is significant
        if abs(alpha2_new - alpha2_old) < 1e-5:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new
        
        return True
    
    def fit(self, X: SparseMatrix, y: List[int]) -> 'SVMClassifier':
        """Train SVM using SMO algorithm."""
        # Convert labels to {-1, +1}
        unique_labels = sorted(set(y))
        if len(unique_labels) != 2:
            raise ValueError("SVM supports binary classification only")
        
        # Map labels to {-1, +1}
        self.label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        self.inverse_label_map = {-1: unique_labels[0], 1: unique_labels[1]}
        
        self.X_train = X
        self.y_train = [self.label_map[label] for label in y]
        
        # Initialize alphas
        n_samples = len(self.y_train)
        self.alphas = [0.0] * n_samples
        self.b = 0.0
        
        # Clear kernel cache
        self._kernel_cache.clear()
        
        # Set random seed
        random.seed(self.random_state)
        
        # SMO main loop
        passes = 0
        num_changed = 0
        examine_all = True
        
        while (passes < self.max_passes) and (num_changed > 0 or examine_all):
            num_changed = 0
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Examine non-bound samples only
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            passes += 1
        
        # Update threshold
        self._update_threshold()
        
        # Store support vectors
        self._store_support_vectors()
        
        self.is_fitted = True
        return self
    
    def _examine_example(self, i1: int) -> int:
        """Examine example i1 for possible optimization."""
        y1 = self.y_train[i1]
        alpha1 = self.alphas[i1]
        E1 = self._compute_error(i1)
        r1 = E1 * y1
        
        # Check KKT conditions
        if ((r1 < -self.tol and alpha1 < self.C) or 
            (r1 > self.tol and alpha1 > 0)):
            
            # Try to find second alpha
            # First try non-bound examples
            non_bound_indices = [i for i, alpha in enumerate(self.alphas) 
                               if 0 < alpha < self.C and i != i1]
            
            if non_bound_indices:
                i2 = self._select_second_alpha(i1, E1)
                if self._smo_step(i1, i2):
                    return 1
            
            # Try all examples as second alpha
            indices = list(range(len(self.alphas)))
            random.shuffle(indices)
            
            for i2 in indices:
                if i2 != i1 and self._smo_step(i1, i2):
                    return 1
        
        return 0
    
    def _store_support_vectors(self) -> None:
        """Store support vector information."""
        self.support_vectors_ = []
        self.support_vector_alphas_ = []
        self.support_vector_labels_ = []
        
        for i, alpha in enumerate(self.alphas):
            if alpha > 1e-8:  # Consider as support vector
                self.support_vectors_.append(i)
                self.support_vector_alphas_.append(alpha)
                self.support_vector_labels_.append(self.y_train[i])
    
    def decision_function(self, X: SparseMatrix) -> List[float]:
        """Compute decision function values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        decision_values = []
        
        for query_vector in X.vectors:
            decision_value = 0.0
            
            # Only compute for support vectors (optimization)
            for i in self.support_vectors_:
                alpha_i = self.alphas[i]
                y_i = self.y_train[i]
                
                # Compute kernel between query and support vector
                if self.kernel_type == "linear":
                    kernel_val = query_vector.dot(self.X_train[i])
                elif self.kernel_type == "rbf":
                    # RBF kernel with query vector
                    xi = self.X_train[i]
                    norm_query_sq = sum(v * v for v in query_vector.values)
                    norm_xi_sq = sum(v * v for v in xi.values)
                    dot_product = query_vector.dot(xi)
                    
                    distance_sq = norm_query_sq + norm_xi_sq - 2 * dot_product
                    distance_sq = max(0.0, distance_sq)
                    
                    kernel_val = math.exp(-self.gamma * distance_sq)
                
                decision_value += alpha_i * y_i * kernel_val
            
            decision_value += self.b
            decision_values.append(decision_value)
        
        return decision_values
    
    def predict(self, X: SparseMatrix) -> List[int]:
        """Predict class labels."""
        decision_values = self.decision_function(X)
        # Convert {-1, +1} back to original labels
        predictions = []
        for dv in decision_values:
            svm_label = 1 if dv >= 0 else -1
            original_label = self.inverse_label_map[svm_label]
            predictions.append(original_label)
        
        return predictions
    
    def predict_proba(self, X: SparseMatrix) -> List[List[float]]:
        """Predict class probabilities using Platt scaling approximation."""
        decision_values = self.decision_function(X)
        
        probabilities = []
        for dv in decision_values:
            # Simple sigmoid transformation (Platt scaling approximation)
            # P(y=1|x) = 1 / (1 + exp(-A*f(x) - B))
            # We use simplified version: P(y=1|x) = 1 / (1 + exp(-f(x)))
            prob_pos = 1.0 / (1.0 + math.exp(-dv))
            prob_neg = 1.0 - prob_pos
            
            # Map back to original label order
            if self.inverse_label_map[1] == 1:  # Assuming binary labels 0, 1
                probabilities.append([prob_neg, prob_pos])
            else:
                probabilities.append([prob_pos, prob_neg])
        
        return probabilities
    
    def get_support_vector_info(self) -> Dict[str, Any]:
        """Get information about support vectors for explanation."""
        return {
            'n_support_vectors': len(self.support_vectors_),
            'support_vector_indices': self.support_vectors_,
            'support_vector_alphas': self.support_vector_alphas_,
            'support_vector_labels': self.support_vector_labels_,
            'bias': self.b
        }


def cross_validate_svm(
    X: SparseMatrix, 
    y: List[int], 
    C_values: List[float],
    kernels: List[str],
    gamma_values: List[float],
    cv_folds: int = 5,
    seed: int = 42
) -> List[dict]:
    """Cross-validate SVM with different hyperparameters."""
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
    for C in C_values:
        for kernel in kernels:
            gamma_list = gamma_values if kernel == "rbf" else [0.1]  # Gamma only matters for RBF
            
            for gamma in gamma_list:
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
                    try:
                        svm = SVMClassifier(C=C, kernel=kernel, gamma=gamma, random_state=seed)
                        svm.fit(X_train_fold, y_train_fold)
                        y_pred = svm.predict(X_test_fold)
                        
                        # Compute accuracy
                        correct = sum(1 for true, pred in zip(y_test_fold, y_pred) if true == pred)
                        accuracy = correct / len(y_test_fold) if y_test_fold else 0.0
                        fold_scores.append(accuracy)
                    except Exception as e:
                        print(f"Error in SVM training: {e}")
                        fold_scores.append(0.0)  # Failed training
                
                # Average across folds
                mean_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
                std_score = 0.0
                if len(fold_scores) > 1:
                    variance = sum((score - mean_score) ** 2 for score in fold_scores) / len(fold_scores)
                    std_score = math.sqrt(variance)
                
                result = {
                    'C': C,
                    'kernel': kernel,
                    'gamma': gamma,
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score,
                    'fold_scores': fold_scores
                }
                
                # Only include gamma in results for RBF kernel
                if kernel != "rbf":
                    result.pop('gamma')
                
                results.append(result)
    
    # Sort by mean CV score
    results.sort(key=lambda x: x['mean_cv_score'], reverse=True)
    return results