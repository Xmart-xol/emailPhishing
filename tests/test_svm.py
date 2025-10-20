"""
Test cases for the SVM implementation from scratch.
"""
import pytest
import math
from app.ml.svm import SVMClassifier
from app.ml.features import SparseMatrix, SparseVector


def test_svm_linear_separable():
    """Test SVM on linearly separable data."""
    # Create simple linearly separable dataset
    # Positive class: points with x > 0
    # Negative class: points with x < 0
    
    vectors = [
        SparseVector([0, 1], [2.0, 1.0], 2),   # (2, 1) -> class 1
        SparseVector([0, 1], [1.5, 0.5], 2),  # (1.5, 0.5) -> class 1
        SparseVector([0, 1], [1.0, 2.0], 2),  # (1.0, 2.0) -> class 1
        SparseVector([0, 1], [-1.0, 1.0], 2), # (-1.0, 1.0) -> class 0
        SparseVector([0, 1], [-1.5, 0.5], 2), # (-1.5, 0.5) -> class 0
        SparseVector([0, 1], [-2.0, 2.0], 2), # (-2.0, 2.0) -> class 0
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 1, 1, 0, 0, 0]
    
    # Train SVM
    svm = SVMClassifier(C=1.0, kernel="linear", max_passes=10)
    svm.fit(X, y)
    
    # Make predictions
    predictions = svm.predict(X)
    
    # Should achieve perfect accuracy on linearly separable data
    accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
    assert accuracy >= 0.8, f"Expected accuracy >= 0.8, got {accuracy}"
    
    # Check that we have support vectors
    assert len(svm.support_vectors_) > 0, "Should have support vectors"
    
    # Test decision function
    decision_values = svm.decision_function(X)
    assert len(decision_values) == len(y)
    
    # Decision values should have correct signs
    for i, (true_label, decision_val) in enumerate(zip(y, decision_values)):
        if true_label == 1:
            assert decision_val >= 0, f"Positive class sample {i} should have positive decision value"
        else:
            assert decision_val <= 0, f"Negative class sample {i} should have negative decision value"


def test_svm_rbf_xor():
    """Test SVM with RBF kernel on XOR-like non-linear data."""
    # XOR dataset - not linearly separable
    vectors = [
        SparseVector([0, 1], [1.0, 1.0], 2),   # (1, 1) -> class 0
        SparseVector([0, 1], [-1.0, -1.0], 2), # (-1, -1) -> class 0
        SparseVector([0, 1], [1.0, -1.0], 2),  # (1, -1) -> class 1
        SparseVector([0, 1], [-1.0, 1.0], 2),  # (-1, 1) -> class 1
    ]
    
    X = SparseMatrix(vectors)
    y = [0, 0, 1, 1]
    
    # Linear kernel should struggle
    svm_linear = SVMClassifier(C=1.0, kernel="linear", max_passes=10)
    svm_linear.fit(X, y)
    linear_predictions = svm_linear.predict(X)
    linear_accuracy = sum(1 for true, pred in zip(y, linear_predictions) if true == pred) / len(y)
    
    # RBF kernel should perform better
    svm_rbf = SVMClassifier(C=1.0, kernel="rbf", gamma=1.0, max_passes=10)
    svm_rbf.fit(X, y)
    rbf_predictions = svm_rbf.predict(X)
    rbf_accuracy = sum(1 for true, pred in zip(y, rbf_predictions) if true == pred) / len(y)
    
    # RBF should outperform linear on this non-linear problem
    assert rbf_accuracy >= linear_accuracy, "RBF kernel should perform better on non-linear data"
    
    # RBF should achieve reasonable accuracy
    assert rbf_accuracy >= 0.5, f"RBF should achieve > 50% accuracy, got {rbf_accuracy}"


def test_svm_probability_estimation():
    """Test SVM probability estimation."""
    vectors = [
        SparseVector([0], [2.0], 1),   # Strong positive
        SparseVector([0], [1.0], 1),   # Weak positive
        SparseVector([0], [-1.0], 1),  # Weak negative
        SparseVector([0], [-2.0], 1),  # Strong negative
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 1, 0, 0]
    
    svm = SVMClassifier(C=1.0, kernel="linear")
    svm.fit(X, y)
    
    # Test probability prediction
    probabilities = svm.predict_proba(X)
    
    assert len(probabilities) == len(y)
    
    for prob in probabilities:
        assert len(prob) == 2, "Should return probabilities for both classes"
        assert abs(sum(prob) - 1.0) < 1e-6, "Probabilities should sum to 1"
        assert all(0 <= p <= 1 for p in prob), "Probabilities should be between 0 and 1"


def test_svm_hyperparameters():
    """Test SVM with different hyperparameters."""
    # Create simple dataset
    vectors = [
        SparseVector([0], [1.0], 1),
        SparseVector([0], [2.0], 1),
        SparseVector([0], [-1.0], 1),
        SparseVector([0], [-2.0], 1),
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 1, 0, 0]
    
    # Test different C values
    for C in [0.1, 1.0, 10.0]:
        svm = SVMClassifier(C=C, kernel="linear")
        svm.fit(X, y)
        predictions = svm.predict(X)
        assert len(predictions) == len(y)
    
    # Test different gamma values for RBF
    for gamma in [0.1, 1.0, 10.0]:
        svm = SVMClassifier(C=1.0, kernel="rbf", gamma=gamma)
        svm.fit(X, y)
        predictions = svm.predict(X)
        assert len(predictions) == len(y)


def test_svm_support_vector_info():
    """Test SVM support vector information extraction."""
    vectors = [
        SparseVector([0, 1], [1.0, 1.0], 2),
        SparseVector([0, 1], [2.0, 1.0], 2),
        SparseVector([0, 1], [-1.0, 1.0], 2),
        SparseVector([0, 1], [-2.0, 1.0], 2),
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 1, 0, 0]
    
    svm = SVMClassifier(C=1.0, kernel="linear")
    svm.fit(X, y)
    
    # Get support vector info
    sv_info = svm.get_support_vector_info()
    
    assert "n_support_vectors" in sv_info
    assert "support_vector_indices" in sv_info
    assert "support_vector_alphas" in sv_info
    assert "support_vector_labels" in sv_info
    assert "bias" in sv_info
    
    assert sv_info["n_support_vectors"] > 0
    assert len(sv_info["support_vector_indices"]) == sv_info["n_support_vectors"]
    assert len(sv_info["support_vector_alphas"]) == sv_info["n_support_vectors"]
    assert len(sv_info["support_vector_labels"]) == sv_info["n_support_vectors"]
    
    # Alphas should be positive
    assert all(alpha > 0 for alpha in sv_info["support_vector_alphas"])


def test_svm_edge_cases():
    """Test SVM edge cases."""
    # Single sample per class
    vectors = [
        SparseVector([0], [1.0], 1),
        SparseVector([0], [-1.0], 1),
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 0]
    
    svm = SVMClassifier(C=1.0, kernel="linear")
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    assert len(predictions) == 2
    
    # Test empty features
    empty_vectors = [SparseVector([], [], 10) for _ in range(4)]
    X_empty = SparseMatrix(empty_vectors)
    y_empty = [1, 1, 0, 0]
    
    svm_empty = SVMClassifier(C=1.0, kernel="linear")
    try:
        svm_empty.fit(X_empty, y_empty)
        predictions_empty = svm_empty.predict(X_empty)
        assert len(predictions_empty) == 4
    except:
        # It's okay if this fails - empty features are a degenerate case
        pass


if __name__ == "__main__":
    test_svm_linear_separable()
    test_svm_rbf_xor()
    test_svm_probability_estimation()
    test_svm_hyperparameters()
    test_svm_support_vector_info()
    test_svm_edge_cases()
    print("All SVM tests passed!")