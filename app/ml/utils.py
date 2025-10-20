"""
Utility functions for ML operations.
"""
import random
from typing import List, Tuple, Any, Dict
from .features import SparseMatrix


def train_test_split(
    X: SparseMatrix, 
    y: List[int], 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[SparseMatrix, SparseMatrix, List[int], List[int]]:
    """Split data into train and test sets."""
    random.seed(random_state)
    n_samples = len(X)
    
    if stratify:
        # Group indices by class
        class_indices = {}
        for i, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # Split each class proportionally
        train_indices = []
        test_indices = []
        
        for label, indices in class_indices.items():
            random.shuffle(indices)
            n_test = int(len(indices) * test_size)
            test_indices.extend(indices[:n_test])
            train_indices.extend(indices[n_test:])
        
        # Shuffle the final splits
        random.shuffle(train_indices)
        random.shuffle(test_indices)
    else:
        # Simple random split
        indices = list(range(n_samples))
        random.shuffle(indices)
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    # Create splits
    X_train_vectors = [X[i] for i in train_indices]
    X_test_vectors = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    X_train = SparseMatrix(X_train_vectors)
    X_test = SparseMatrix(X_test_vectors)
    
    return X_train, X_test, y_train, y_test


def k_fold_split(
    X: SparseMatrix, 
    y: List[int], 
    k: int = 5, 
    shuffle: bool = True, 
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Create k-fold cross-validation splits."""
    if shuffle:
        random.seed(random_state)
    
    n_samples = len(X)
    indices = list(range(n_samples))
    
    if shuffle:
        random.shuffle(indices)
    
    fold_size = n_samples // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:  # Last fold gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * fold_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in test_indices]
        
        folds.append((train_indices, test_indices))
    
    return folds


def stratified_k_fold_split(
    X: SparseMatrix, 
    y: List[int], 
    k: int = 5, 
    shuffle: bool = True, 
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Create stratified k-fold cross-validation splits."""
    if shuffle:
        random.seed(random_state)
    
    # Group indices by class
    class_indices = {}
    for i, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Shuffle each class
    if shuffle:
        for indices in class_indices.values():
            random.shuffle(indices)
    
    # Create folds
    folds = [[] for _ in range(k)]
    
    # Distribute each class across folds
    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        remainder = len(indices) % k
        
        start_idx = 0
        for fold_idx in range(k):
            # Some folds get one extra sample if there's remainder
            end_idx = start_idx + fold_size + (1 if fold_idx < remainder else 0)
            folds[fold_idx].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # Convert to train/test splits
    fold_splits = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        
        fold_splits.append((train_indices, test_indices))
    
    return fold_splits


def balance_classes(
    X: SparseMatrix, 
    y: List[int], 
    method: str = "undersample", 
    random_state: int = 42
) -> Tuple[SparseMatrix, List[int]]:
    """Balance classes in the dataset."""
    random.seed(random_state)
    
    # Count class frequencies
    class_counts = {}
    class_indices = {}
    
    for i, label in enumerate(y):
        if label not in class_counts:
            class_counts[label] = 0
            class_indices[label] = []
        class_counts[label] += 1
        class_indices[label].append(i)
    
    if len(class_counts) != 2:
        raise ValueError("Class balancing currently supports binary classification only")
    
    labels = list(class_counts.keys())
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if method == "undersample":
        # Undersample majority class
        target_count = min_count
        balanced_indices = []
        
        for label in labels:
            indices = class_indices[label]
            random.shuffle(indices)
            balanced_indices.extend(indices[:target_count])
    
    elif method == "oversample":
        # Oversample minority class with replacement
        target_count = max_count
        balanced_indices = []
        
        for label in labels:
            indices = class_indices[label]
            current_count = len(indices)
            
            if current_count < target_count:
                # Oversample with replacement
                additional_needed = target_count - current_count
                additional_indices = random.choices(indices, k=additional_needed)
                balanced_indices.extend(indices + additional_indices)
            else:
                balanced_indices.extend(indices)
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Shuffle balanced indices
    random.shuffle(balanced_indices)
    
    # Create balanced dataset
    X_balanced_vectors = [X[i] for i in balanced_indices]
    y_balanced = [y[i] for i in balanced_indices]
    
    X_balanced = SparseMatrix(X_balanced_vectors)
    
    return X_balanced, y_balanced


def grid_search_cv(
    estimator_class: Any,
    param_grid: Dict[str, List[Any]], 
    X: SparseMatrix, 
    y: List[int],
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform grid search with cross-validation."""
    from itertools import product
    from .metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Create CV folds
    cv_folds = stratified_k_fold_split(X, y, k=cv, random_state=random_state)
    
    best_score = -float('inf')
    best_params = None
    results = []
    
    for param_combo in param_combinations:
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        
        # Cross-validation scores
        cv_scores = []
        
        for train_indices, test_indices in cv_folds:
            # Create train/test splits
            X_train_vectors = [X[i] for i in train_indices]
            X_test_vectors = [X[i] for i in test_indices]
            y_train_fold = [y[i] for i in train_indices]
            y_test_fold = [y[i] for i in test_indices]
            
            X_train_fold = SparseMatrix(X_train_vectors)
            X_test_fold = SparseMatrix(X_test_vectors)
            
            try:
                # Train model
                model = estimator_class(**params, random_state=random_state)
                model.fit(X_train_fold, y_train_fold)
                
                # Make predictions
                if scoring == "roc_auc" and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_fold)
                    # Extract positive class probabilities
                    y_scores = [proba[1] for proba in y_proba]
                    score = roc_auc_score(y_test_fold, y_scores)
                else:
                    y_pred = model.predict(X_test_fold)
                    
                    if scoring == "accuracy":
                        score = accuracy_score(y_test_fold, y_pred)
                    elif scoring == "precision":
                        score = precision_score(y_test_fold, y_pred)
                    elif scoring == "recall":
                        score = recall_score(y_test_fold, y_pred)
                    elif scoring == "f1":
                        score = f1_score(y_test_fold, y_pred)
                    else:
                        raise ValueError(f"Unknown scoring method: {scoring}")
                
                cv_scores.append(score)
                
            except Exception as e:
                print(f"Error with params {params}: {e}")
                cv_scores.append(0.0)  # Failed fold
        
        # Compute mean and std of CV scores
        mean_score = sum(cv_scores) / len(cv_scores) if cv_scores else 0.0
        std_score = 0.0
        if len(cv_scores) > 1:
            variance = sum((score - mean_score) ** 2 for score in cv_scores) / len(cv_scores)
            std_score = (variance ** 0.5)
        
        result = {
            'params': params,
            'mean_test_score': mean_score,
            'std_test_score': std_score,
            'cv_scores': cv_scores
        }
        results.append(result)
        
        # Track best parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    # Sort results by score
    results.sort(key=lambda x: x['mean_test_score'], reverse=True)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': results
    }


def compute_class_weights(y: List[int]) -> Dict[int, float]:
    """Compute class weights for imbalanced datasets."""
    # Count class frequencies
    class_counts = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    n_samples = len(y)
    n_classes = len(class_counts)
    
    # Compute balanced weights: n_samples / (n_classes * n_samples_class)
    class_weights = {}
    for label, count in class_counts.items():
        class_weights[label] = n_samples / (n_classes * count)
    
    return class_weights


def feature_importance_permutation(
    model: Any,
    X: SparseMatrix,
    y: List[int],
    scoring: str = "accuracy",
    n_repeats: int = 5,
    random_state: int = 42
) -> List[float]:
    """Compute feature importance using permutation method."""
    from .metrics import accuracy_score
    random.seed(random_state)
    
    # Get baseline score
    y_pred_baseline = model.predict(X)
    if scoring == "accuracy":
        baseline_score = accuracy_score(y, y_pred_baseline)
    else:
        raise ValueError(f"Scoring method {scoring} not implemented")
    
    n_features = X.vectors[0].dim if X.vectors else 0
    feature_importances = [0.0] * n_features
    
    for feature_idx in range(n_features):
        scores_drop = []
        
        for _ in range(n_repeats):
            # Create permuted dataset
            X_permuted_vectors = []
            
            for vector in X.vectors:
                # Copy vector and permute specific feature
                new_indices = vector.indices.copy()
                new_values = vector.values.copy()
                
                # Find if this feature exists in the sparse vector
                if feature_idx in vector.indices:
                    idx_pos = vector.indices.index(feature_idx)
                    # Randomly permute this feature value across all samples
                    all_feature_values = []
                    for v in X.vectors:
                        if feature_idx in v.indices:
                            val_pos = v.indices.index(feature_idx)
                            all_feature_values.append(v.values[val_pos])
                        else:
                            all_feature_values.append(0.0)
                    
                    random.shuffle(all_feature_values)
                    new_values[idx_pos] = all_feature_values[0]  # Use first shuffled value
                
                from .features import SparseVector
                X_permuted_vectors.append(SparseVector(new_indices, new_values, vector.dim))
            
            X_permuted = SparseMatrix(X_permuted_vectors)
            
            # Compute score with permuted feature
            y_pred_permuted = model.predict(X_permuted)
            permuted_score = accuracy_score(y, y_pred_permuted)
            
            # Importance is the drop in performance
            importance = baseline_score - permuted_score
            scores_drop.append(importance)
        
        # Average importance across repeats
        feature_importances[feature_idx] = sum(scores_drop) / len(scores_drop)
    
    return feature_importances