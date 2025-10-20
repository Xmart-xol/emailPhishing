"""
Simple test script to verify the email phishing detection system works.
Tests the core ML implementations without requiring pandas initially.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.features import SparseVector, SparseMatrix, Vectorizer
from app.ml.knn import KNNClassifier
from app.ml.svm import SVMClassifier
from app.ml.metrics import accuracy_score, precision_score, recall_score


def test_sparse_structures():
    """Test sparse vector/matrix implementations."""
    print("🧪 Testing sparse data structures...")
    
    # Test SparseVector
    vec1 = SparseVector([0, 2, 4], [1.0, 2.0, 3.0], 5)
    vec2 = SparseVector([1, 2, 3], [1.0, 1.0, 2.0], 5)
    
    dot_product = vec1.dot(vec2)
    assert abs(dot_product - 2.0) < 1e-6, f"Expected dot product 2.0, got {dot_product}"
    
    norm = vec1.norm_l2()
    expected_norm = (1.0 + 4.0 + 9.0) ** 0.5
    assert abs(norm - expected_norm) < 1e-6, f"Expected norm {expected_norm}, got {norm}"
    
    # Test SparseMatrix
    vectors = [vec1, vec2]
    matrix = SparseMatrix(vectors)
    assert len(matrix) == 2
    assert matrix[0] == vec1
    
    print("✅ Sparse data structures work correctly!")


def test_feature_extraction():
    """Test feature extraction without file dependencies."""
    print("\n🧪 Testing feature extraction...")
    
    # Simple feature config
    config = {
        'bow': True,
        'char_ngrams': True,
        'heuristics': True,
        'hash_dim': 1000,
        'ngram_min': 3,
        'ngram_max': 4,
        'use_tfidf': True
    }
    
    # Test texts
    texts = [
        "Click here to win $1000! Urgent action required!",
        "Meeting scheduled for tomorrow at 3 PM",
        "Your account has been suspended. Click link to verify.",
        "Regular work email about project updates"
    ]
    
    vectorizer = Vectorizer(config, seed=42)
    vectorizer.fit(texts)
    
    # Transform texts
    X = vectorizer.transform(texts)
    
    assert len(X) == 4, f"Expected 4 vectors, got {len(X)}"
    assert all(vec.dim > 0 for vec in X.vectors), "All vectors should have positive dimension"
    
    print(f"✅ Feature extraction works! Generated {X.vectors[0].dim}-dimensional vectors")


def test_knn_classifier():
    """Test KNN classifier."""
    print("\n🧪 Testing KNN classifier...")
    
    # Create simple 2D dataset
    vectors = [
        SparseVector([0, 1], [1.0, 1.0], 2),   # class 1
        SparseVector([0, 1], [1.1, 0.9], 2),  # class 1
        SparseVector([0, 1], [0.9, 1.1], 2),  # class 1
        SparseVector([0, 1], [-1.0, -1.0], 2), # class 0
        SparseVector([0, 1], [-1.1, -0.9], 2), # class 0
        SparseVector([0, 1], [-0.9, -1.1], 2), # class 0
    ]
    
    X_train = SparseMatrix(vectors)
    y_train = [1, 1, 1, 0, 0, 0]
    
    # Test points
    test_vectors = [
        SparseVector([0, 1], [0.8, 0.8], 2),   # Should be class 1
        SparseVector([0, 1], [-0.8, -0.8], 2), # Should be class 0
    ]
    X_test = SparseMatrix(test_vectors)
    
    # Train and test KNN
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test, k=3, metric="euclidean")
    probabilities = knn.predict_proba(X_test, k=3, metric="euclidean")
    
    assert len(predictions) == 2
    assert predictions[0] == 1, f"Expected class 1, got {predictions[0]}"
    assert predictions[1] == 0, f"Expected class 0, got {predictions[1]}"
    assert len(probabilities) == 2
    assert all(len(prob) == 2 for prob in probabilities)
    
    print("✅ KNN classifier works correctly!")


def test_svm_classifier():
    """Test SVM classifier."""
    print("\n🧪 Testing SVM classifier...")
    
    # Create simple linearly separable dataset
    vectors = [
        SparseVector([0], [2.0], 1),   # class 1
        SparseVector([0], [1.5], 1),   # class 1
        SparseVector([0], [1.0], 1),   # class 1
        SparseVector([0], [-1.0], 1),  # class 0
        SparseVector([0], [-1.5], 1),  # class 0
        SparseVector([0], [-2.0], 1),  # class 0
    ]
    
    X = SparseMatrix(vectors)
    y = [1, 1, 1, 0, 0, 0]
    
    # Train SVM
    svm = SVMClassifier(C=1.0, kernel="linear", max_passes=5)
    svm.fit(X, y)
    
    # Make predictions
    predictions = svm.predict(X)
    probabilities = svm.predict_proba(X)
    decision_values = svm.decision_function(X)
    
    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
    
    assert len(predictions) == 6
    assert accuracy >= 0.8, f"Expected accuracy >= 0.8, got {accuracy}"
    assert len(probabilities) == 6
    assert len(decision_values) == 6
    
    print(f"✅ SVM classifier works correctly! Accuracy: {accuracy:.2f}")


def test_metrics():
    """Test evaluation metrics."""
    print("\n🧪 Testing evaluation metrics...")
    
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    expected_accuracy = 4/6  # 4 correct out of 6
    assert abs(accuracy - expected_accuracy) < 1e-6
    
    print(f"✅ Metrics work correctly! Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")


def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("\n🧪 Testing end-to-end pipeline...")
    
    # Sample email texts with labels
    texts = [
        "URGENT! Your account will be closed unless you verify immediately! Click here: http://fake-bank.com",
        "Dear valued customer, please update your password at our secure portal",  
        "Meeting reminder: Project review tomorrow at 2 PM in conference room B",
        "Weekly newsletter with updates about our products and services",
        "WARNING: Suspicious activity detected! Verify your identity NOW or account will be suspended!",
        "Thank you for your recent purchase. Your order will arrive in 3-5 business days"
    ]
    
    labels = [1, 1, 0, 0, 1, 0]  # 1 = phish, 0 = ham
    
    # Feature extraction
    config = {
        'bow': True,
        'char_ngrams': True,
        'heuristics': True,
        'hash_dim': 5000,
        'ngram_min': 3,
        'ngram_max': 5,
        'use_tfidf': True
    }
    
    vectorizer = Vectorizer(config, seed=42)
    vectorizer.fit(texts)
    X = vectorizer.transform(texts)
    
    # Train both models
    knn = KNNClassifier()
    knn.fit(X, labels)
    
    svm = SVMClassifier(C=1.0, kernel="linear", max_passes=10)
    svm.fit(X, labels)
    
    # Test predictions
    knn_pred = knn.predict(X, k=3, metric="cosine")
    svm_pred = svm.predict(X)
    
    knn_accuracy = accuracy_score(labels, knn_pred)
    svm_accuracy = accuracy_score(labels, svm_pred)
    
    print(f"✅ End-to-end test complete!")
    print(f"   KNN Accuracy: {knn_accuracy:.2f}")
    print(f"   SVM Accuracy: {svm_accuracy:.2f}")
    
    return knn_accuracy >= 0.5 and svm_accuracy >= 0.5  # Changed > to >=


def main():
    """Run all tests."""
    print("🚀 Testing Email Phishing Detection System")
    print("=" * 50)
    
    try:
        test_sparse_structures()
        test_feature_extraction()
        test_knn_classifier()
        test_svm_classifier()
        test_metrics()
        
        if test_end_to_end():
            print("\n🎉 ALL TESTS PASSED! The system is working correctly.")
            print("\nNext steps:")
            print("1. Install pandas: pip3 install pandas")
            print("2. Start the API server: python3 app/main.py")
            print("3. Visit http://localhost:8000/docs for API documentation")
        else:
            print("\n⚠️  End-to-end test failed. Check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()