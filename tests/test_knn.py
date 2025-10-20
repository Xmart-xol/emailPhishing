"""
Test cases for the KNN implementation from scratch.
"""
import pytest
import math
from app.ml.knn import KNNClassifier
from app.ml.features import SparseMatrix, SparseVector


def test_knn_basic_classification():
    """Test basic KNN classification."""
    # Create simple 2D dataset
    vectors = [
        SparseVector([0, 1], [1.0, 1.0], 2),   # (1, 1) -> class 1
        SparseVector([0, 1], [1.1, 0.9], 2),  # (1.1, 0.9) -> class 1
        SparseVector([0, 1], [0.9, 1.1], 2),  # (0.9, 1.1) -> class 1
        SparseVector([0, 1], [-1.0, -1.0], 2), # (-1, -1) -> class 0
        SparseVector([0, 1], [-1.1, -0.9], 2), # (-1.1, -0.9) -> class 0
        SparseVector([0, 1], [-0.9, -1.1], 2), # (-0.9, -1.1) -> class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 1, 1, 0, 0, 0]
    
    # Test points
    test_vectors = [
        SparseVector([0, 1], [0.8, 0.8], 2),   # Should be class 1
        SparseVector([0, 1], [-0.8, -0.8], 2), # Should be class 0
    ]
    X_test = SparseMatrix(test_vectors)
    
    # Train KNN
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Make predictions with k=3
    predictions = knn.predict(X_test, k=3, metric="euclidean")
    
    assert len(predictions) == 2
    assert predictions[0] == 1, "Point (0.8, 0.8) should be classified as class 1"
    assert predictions[1] == 0, "Point (-0.8, -0.8) should be classified as class 0"


def test_knn_cosine_vs_euclidean():
    """Test KNN with different distance metrics."""
    # Create vectors where cosine and Euclidean might give different results
    vectors = [
        SparseVector([0, 1], [1.0, 0.0], 2),   # (1, 0) -> class 1
        SparseVector([0, 1], [0.0, 1.0], 2),   # (0, 1) -> class 1
        SparseVector([0, 1], [-1.0, 0.0], 2),  # (-1, 0) -> class 0
        SparseVector([0, 1], [0.0, -1.0], 2),  # (0, -1) -> class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 1, 0, 0]
    
    # Test point
    test_vectors = [SparseVector([0, 1], [0.7, 0.7], 2)]  # (0.7, 0.7)
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Test both metrics
    pred_euclidean = knn.predict(X_test, k=2, metric="euclidean")
    pred_cosine = knn.predict(X_test, k=2, metric="cosine")
    
    assert len(pred_euclidean) == 1
    assert len(pred_cosine) == 1
    
    # Both should classify as class 1 (closer to positive quadrant)
    assert pred_euclidean[0] == 1
    assert pred_cosine[0] == 1


def test_knn_weighting():
    """Test KNN with uniform vs distance weighting."""
    # Create dataset where distance weighting might make a difference
    vectors = [
        SparseVector([0], [1.0], 1),   # Very close to test point -> class 1
        SparseVector([0], [5.0], 1),   # Far from test point -> class 0
        SparseVector([0], [6.0], 1),   # Far from test point -> class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 0, 0]
    
    # Test point close to first training point
    test_vectors = [SparseVector([0], [1.1], 1)]
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # With uniform weighting, might be influenced by majority
    pred_uniform = knn.predict(X_test, k=3, weighting="uniform")
    
    # With distance weighting, closest point should dominate
    pred_distance = knn.predict(X_test, k=3, weighting="distance")
    
    assert len(pred_uniform) == 1
    assert len(pred_distance) == 1
    
    # Distance weighting should favor the closest point (class 1)
    assert pred_distance[0] == 1


def test_knn_probability_estimation():
    """Test KNN probability estimation."""
    vectors = [
        SparseVector([0], [1.0], 1),   # class 1
        SparseVector([0], [1.1], 1),   # class 1
        SparseVector([0], [0.9], 1),   # class 1
        SparseVector([0], [-1.0], 1),  # class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 1, 1, 0]
    
    test_vectors = [SparseVector([0], [1.05], 1)]  # Close to class 1 cluster
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Test probability prediction
    probabilities = knn.predict_proba(X_test, k=3)
    
    assert len(probabilities) == 1
    assert len(probabilities[0]) == 2  # Two classes
    
    prob = probabilities[0]
    assert abs(sum(prob) - 1.0) < 1e-6, "Probabilities should sum to 1"
    assert all(0 <= p <= 1 for p in prob), "Probabilities should be between 0 and 1"
    
    # Should favor class 1 (index 1) since test point is closer to class 1 samples
    assert prob[1] > prob[0], "Should have higher probability for class 1"


def test_knn_k_values():
    """Test KNN with different k values."""
    # Create imbalanced dataset
    vectors = [
        SparseVector([0], [1.0], 1),   # class 1
        SparseVector([0], [0.0], 1),   # class 0
        SparseVector([0], [-0.1], 1),  # class 0
        SparseVector([0], [-0.2], 1),  # class 0
        SparseVector([0], [-0.3], 1),  # class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 0, 0, 0, 0]
    
    test_vectors = [SparseVector([0], [0.1], 1)]  # Between the classes
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Test different k values
    for k in [1, 3, 5]:
        predictions = knn.predict(X_test, k=k)
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    # k=1 should be influenced by closest neighbor
    pred_k1 = knn.predict(X_test, k=1)[0]
    
    # k=5 (all neighbors) should be influenced by majority class (0)
    pred_k5 = knn.predict(X_test, k=5)[0]
    assert pred_k5 == 0, "With k=5, majority class (0) should win"


def test_knn_neighbors_explanation():
    """Test KNN neighbor explanation functionality."""
    vectors = [
        SparseVector([0], [1.0], 1),   # class 1
        SparseVector([0], [2.0], 1),   # class 1
        SparseVector([0], [-1.0], 1),  # class 0
        SparseVector([0], [-2.0], 1),  # class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 1, 0, 0]
    
    test_vector = SparseVector([0], [0.5], 1)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Get neighbors for explanation
    neighbors = knn.get_neighbors_for_explanation(test_vector, k=3)
    
    assert len(neighbors) == 3
    
    for idx, distance, label in neighbors:
        assert 0 <= idx < len(y_train)
        assert distance >= 0
        assert label in [0, 1]
    
    # Neighbors should be sorted by distance (closest first)
    distances = [distance for _, distance, _ in neighbors]
    assert distances == sorted(distances), "Neighbors should be sorted by distance"


def test_knn_edge_cases():
    """Test KNN edge cases."""
    # Minimum dataset
    vectors = [
        SparseVector([0], [1.0], 1),   # class 1
        SparseVector([0], [-1.0], 1),  # class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 0]
    
    test_vectors = [SparseVector([0], [0.0], 1)]
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    # Test with k larger than training set
    predictions = knn.predict(X_test, k=5)  # k > n_samples
    assert len(predictions) == 1
    
    # Test with empty features
    empty_vectors = [SparseVector([], [], 10) for _ in range(4)]
    X_empty = SparseMatrix(empty_vectors)
    y_empty = [1, 1, 0, 0]
    
    knn_empty = KNNClassifier()
    knn_empty.fit(X_empty, y_empty)
    
    test_empty = [SparseVector([], [], 10)]
    X_test_empty = SparseMatrix(test_empty)
    
    # Should handle empty features gracefully
    predictions_empty = knn_empty.predict(X_test_empty, k=2)
    assert len(predictions_empty) == 1


def test_knn_performance():
    """Test KNN computational aspects (not strict performance test)."""
    # Create larger dataset to test basic scalability
    import random
    random.seed(42)
    
    vectors = []
    labels = []
    
    for i in range(100):
        # Create random 2D points
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        
        # Simple classification: positive if x + y > 0
        label = 1 if x + y > 0 else 0
        
        vectors.append(SparseVector([0, 1], [x, y], 2))
        labels.append(label)
    
    X_train = SparseMatrix(vectors)
    y_train = labels
    
    # Test points
    test_vectors = [SparseVector([0, 1], [0.5, 0.5], 2) for _ in range(10)]
    X_test = SparseMatrix(test_vectors)
    
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    import time
    start_time = time.time()
    predictions = knn.predict(X_test, k=5)
    end_time = time.time()
    
    assert len(predictions) == 10
    
    # Should complete reasonably quickly (not a strict requirement)
    elapsed = end_time - start_time
    assert elapsed < 5.0, f"KNN took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    test_knn_basic_classification()
    test_knn_cosine_vs_euclidean()
    test_knn_weighting()
    test_knn_probability_estimation()
    test_knn_k_values()
    test_knn_neighbors_explanation()
    test_knn_edge_cases()
    test_knn_performance()
    print("All KNN tests passed!")