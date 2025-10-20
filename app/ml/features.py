"""
Feature extraction for email phishing detection.
Implements BoW, char n-grams, and heuristic features from scratch.
"""
import re
import hashlib
from typing import Dict, List, Tuple, TypedDict, Optional, Any
from collections import defaultdict
import math


class FeatureConfig(TypedDict):
    bow: bool
    char_ngrams: bool
    heuristics: bool
    hash_dim: int
    ngram_min: int
    ngram_max: int
    use_tfidf: bool


class EnhancedFeatureConfig(TypedDict):
    bow: bool
    char_ngrams: bool
    heuristics: bool
    hash_dim: int
    ngram_min: int
    ngram_max: int
    use_tfidf: bool
    # New fields for rich datasets
    use_sender_features: bool
    use_temporal_features: bool
    use_structural_features: bool


class SparseVector:
    """Sparse vector implementation for memory efficiency."""
    
    def __init__(self, indices: List[int] = None, values: List[float] = None, dim: int = 0):
        self.indices = indices or []
        self.values = values or []
        self.dim = dim
    
    def dot(self, other: 'SparseVector') -> float:
        """Compute dot product with another sparse vector."""
        result = 0.0
        i = j = 0
        while i < len(self.indices) and j < len(other.indices):
            if self.indices[i] == other.indices[j]:
                result += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif self.indices[i] < other.indices[j]:
                i += 1
            else:
                j += 1
        return result
    
    def norm_l2(self) -> float:
        """Compute L2 norm."""
        return math.sqrt(sum(v * v for v in self.values))
    
    def normalize_l2_inplace(self) -> 'SparseVector':
        """Normalize vector in-place."""
        norm = self.norm_l2()
        if norm > 0:
            self.values = [v / norm for v in self.values]
        return self
    
    def to_dense(self) -> List[float]:
        """Convert to dense vector."""
        dense = [0.0] * self.dim
        for idx, val in zip(self.indices, self.values):
            if 0 <= idx < self.dim:
                dense[idx] = val
        return dense


class SparseMatrix:
    """Sparse matrix for storing multiple vectors."""
    
    def __init__(self, vectors: List[SparseVector] = None):
        self.vectors = vectors or []
    
    def __len__(self) -> int:
        return len(self.vectors)
    
    def __getitem__(self, idx):
        """Get item(s) by index - supports both single indices and lists/slices."""
        if isinstance(idx, int):
            return self.vectors[idx]
        elif isinstance(idx, (list, tuple)):
            # Return a new SparseMatrix with the selected vectors
            selected_vectors = [self.vectors[i] for i in idx]
            return SparseMatrix(selected_vectors)
        elif isinstance(idx, slice):
            # Return a new SparseMatrix with the sliced vectors
            selected_vectors = self.vectors[idx]
            return SparseMatrix(selected_vectors)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def append(self, vector: SparseVector):
        self.vectors.append(vector)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of the matrix (n_samples, n_features)."""
        n_samples = len(self.vectors)
        n_features = self.vectors[0].dim if self.vectors else 0
        return (n_samples, n_features)


def stable_hash(text: str, seed: int = 42) -> int:
    """Stable hash function using hashlib for reproducibility."""
    hasher = hashlib.md5()
    hasher.update(f"{seed}{text}".encode('utf-8'))
    return int.from_bytes(hasher.digest()[:4], 'big', signed=False)


class BagOfWordsVectorizer:
    """Bag of Words vectorizer with hashing trick."""
    
    def __init__(self, hash_dim: int = 20000, seed: int = 42):
        self.hash_dim = hash_dim
        self.seed = seed
        self.idf_weights: Dict[int, float] = {}
        self.use_tfidf = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-word characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 2]  # Filter short tokens
    
    def _hash_token(self, token: str) -> int:
        """Hash token to index."""
        return stable_hash(token, self.seed) % self.hash_dim
    
    def fit(self, texts: List[str], use_tfidf: bool = True) -> 'BagOfWordsVectorizer':
        """Fit IDF weights if using TF-IDF."""
        self.use_tfidf = use_tfidf
        if not use_tfidf:
            return self
        
        # Count document frequencies
        df_counts = defaultdict(int)
        n_docs = len(texts)
        
        for text in texts:
            tokens = self._tokenize(text)
            unique_hashes = set(self._hash_token(token) for token in tokens)
            for hash_idx in unique_hashes:
                df_counts[hash_idx] += 1
        
        # Compute IDF weights
        for hash_idx, df in df_counts.items():
            # Add smoothing to prevent division by zero
            idf = math.log(n_docs / (df + 1)) + 1.0
            self.idf_weights[hash_idx] = max(idf, 0.1)  # Clip IDF
        
        return self
    
    def transform(self, texts: List[str]) -> SparseMatrix:
        """Transform texts to sparse vectors."""
        vectors = []
        
        for text in texts:
            tokens = self._tokenize(text)
            tf_counts = defaultdict(int)
            
            # Count term frequencies
            for token in tokens:
                hash_idx = self._hash_token(token)
                tf_counts[hash_idx] += 1
            
            # Convert to sparse vector
            indices = []
            values = []
            
            for hash_idx, tf in tf_counts.items():
                if tf > 0:
                    if self.use_tfidf:
                        # TF-IDF with log normalization
                        tf_weight = math.log(1 + tf)
                        idf_weight = self.idf_weights.get(hash_idx, 1.0)
                        weight = tf_weight * idf_weight
                    else:
                        weight = float(tf)
                    
                    indices.append(hash_idx)
                    values.append(weight)
            
            # Sort by indices for efficient operations
            sorted_pairs = sorted(zip(indices, values))
            if sorted_pairs:
                indices, values = zip(*sorted_pairs)
                vector = SparseVector(list(indices), list(values), self.hash_dim)
            else:
                vector = SparseVector([], [], self.hash_dim)
            
            vectors.append(vector)
        
        return SparseMatrix(vectors)


class CharNGramVectorizer:
    """Character n-gram vectorizer with hashing."""
    
    def __init__(self, ngram_min: int = 3, ngram_max: int = 5, hash_dim: int = 20000, seed: int = 42):
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.hash_dim = hash_dim
        self.seed = seed
    
    def _extract_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams."""
        # Clean text but preserve structure
        text = text.lower()
        ngrams = []
        
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                if ngram.strip():  # Only non-empty n-grams
                    ngrams.append(ngram)
        
        return ngrams
    
    def _hash_ngram(self, ngram: str) -> int:
        """Hash n-gram to index."""
        return stable_hash(ngram, self.seed + 1) % self.hash_dim
    
    def transform(self, texts: List[str]) -> SparseMatrix:
        """Transform texts to sparse vectors."""
        vectors = []
        
        for text in texts:
            ngrams = self._extract_ngrams(text)
            ngram_counts = defaultdict(int)
            
            # Count n-gram frequencies
            for ngram in ngrams:
                hash_idx = self._hash_ngram(ngram)
                ngram_counts[hash_idx] += 1
            
            # Convert to sparse vector
            indices = []
            values = []
            
            for hash_idx, count in ngram_counts.items():
                if count > 0:
                    indices.append(hash_idx)
                    values.append(float(count))
            
            # Sort by indices
            sorted_pairs = sorted(zip(indices, values))
            if sorted_pairs:
                indices, values = zip(*sorted_pairs)
                vector = SparseVector(list(indices), list(values), self.hash_dim)
            else:
                vector = SparseVector([], [], self.hash_dim)
            
            vectors.append(vector)
        
        return SparseMatrix(vectors)


class HeuristicVectorizer:
    """Extract heuristic features from email text."""
    
    def __init__(self):
        self.feature_names = [
            'num_urls', 'has_ip_url', 'num_dots_in_urls', 'suspicious_tld',
            'pct_uppercase', 'num_exclamations', 'num_questions', 'num_currency_symbols',
            'has_urgent_words', 'num_suspicious_words', 'avg_word_length', 'num_html_tags',
            'num_links', 'has_javascript', 'num_recipients', 'num_digits'
        ]
        self.dim = len(self.feature_names)
        
        # Suspicious patterns
        self.suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.cc', '.pw', '.top'}
        self.urgent_words = {'urgent', 'immediate', 'expire', 'limited', 'act now', 'verify', 'suspended', 'click here'}
        self.suspicious_words = {'winner', 'lottery', 'prize', 'free', 'congratulations', 'claim', 'inheritance', 'prince'}
    
    def _count_urls(self, text: str) -> Tuple[int, bool, int]:
        """Count URLs and detect IP addresses."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        num_urls = len(urls)
        has_ip = any(re.search(r'\d+\.\d+\.\d+\.\d+', url) for url in urls)
        num_dots = sum(url.count('.') for url in urls)
        
        return num_urls, has_ip, num_dots
    
    def _has_suspicious_tld(self, text: str) -> bool:
        """Check for suspicious TLDs."""
        return any(tld in text.lower() for tld in self.suspicious_tlds)
    
    def _count_patterns(self, text: str) -> Dict[str, float]:
        """Count various text patterns."""
        features = {}
        
        # Basic character statistics
        if len(text) > 0:
            features['pct_uppercase'] = sum(1 for c in text if c.isupper()) / len(text)
        else:
            features['pct_uppercase'] = 0.0
        
        features['num_exclamations'] = text.count('!')
        features['num_questions'] = text.count('?')
        features['num_currency_symbols'] = text.count('$') + text.count('€') + text.count('£')
        
        # Word-based features
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0.0
        
        # Suspicious content
        features['has_urgent_words'] = float(any(word in text.lower() for word in self.urgent_words))
        features['num_suspicious_words'] = sum(1 for word in self.suspicious_words if word in text.lower())
        
        # HTML/technical features
        features['num_html_tags'] = len(re.findall(r'<[^>]+>', text))
        features['num_links'] = text.lower().count('href=') + text.lower().count('[link]')
        features['has_javascript'] = float('javascript:' in text.lower() or '<script' in text.lower())
        
        # Email-specific features
        features['num_recipients'] = text.count('@')
        features['num_digits'] = sum(1 for c in text if c.isdigit())
        
        return features
    
    def transform(self, texts: List[str]) -> SparseMatrix:
        """Transform texts to heuristic feature vectors."""
        vectors = []
        
        for text in texts:
            # URL features
            num_urls, has_ip, num_dots = self._count_urls(text)
            
            # Other patterns
            pattern_features = self._count_patterns(text)
            
            # Build feature vector
            values = [
                float(num_urls),
                float(has_ip),
                float(num_dots),
                float(self._has_suspicious_tld(text)),
                pattern_features['pct_uppercase'],
                float(pattern_features['num_exclamations']),
                float(pattern_features['num_questions']),
                float(pattern_features['num_currency_symbols']),
                pattern_features['has_urgent_words'],
                float(pattern_features['num_suspicious_words']),
                pattern_features['avg_word_length'],
                float(pattern_features['num_html_tags']),
                float(pattern_features['num_links']),
                pattern_features['has_javascript'],
                float(pattern_features['num_recipients']),
                float(pattern_features['num_digits'])
            ]
            
            # Create dense representation as all features are typically present
            indices = list(range(self.dim))
            vector = SparseVector(indices, values, self.dim)
            vectors.append(vector)
        
        return SparseMatrix(vectors)


class EmailMetadataVectorizer:
    """Extract features from email metadata (sender, receiver, date, etc.)."""
    
    def __init__(self):
        self.sender_vocab = {}
        self.domain_vocab = {}
        
    def _extract_domain(self, email: str) -> str:
        """Extract domain from email address."""
        if '@' in str(email):
            return str(email).split('@')[-1].lower()
        return 'unknown'
    
    def _extract_sender_features(self, sender: str, receiver: str) -> Dict[str, float]:
        """Extract sender-based features."""
        features = {}
        
        sender_str = str(sender).lower() if sender else ''
        receiver_str = str(receiver).lower() if receiver else ''
        
        # Domain features
        sender_domain = self._extract_domain(sender_str)
        receiver_domain = self._extract_domain(receiver_str)
        
        features['sender_domain_suspicious'] = float(
            sender_domain in ['.tk', '.ml', '.ga', '.cf', '.cc', '.pw', '.top']
        )
        features['sender_has_digits'] = float(any(c.isdigit() for c in sender_str))
        features['sender_length'] = len(sender_str)
        features['domain_mismatch'] = float(sender_domain != receiver_domain)
        features['sender_has_noreply'] = float('noreply' in sender_str or 'no-reply' in sender_str)
        
        return features
    
    def _extract_temporal_features(self, date: str) -> Dict[str, float]:
        """Extract time-based features."""
        features = {}
        
        try:
            import datetime
            # Try to parse date (handle multiple formats)
            dt = None
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    dt = datetime.datetime.strptime(str(date), fmt)
                    break
                except:
                    continue
            
            if dt:
                features['hour_of_day'] = dt.hour / 24.0  # Normalize to 0-1
                features['day_of_week'] = dt.weekday() / 7.0  # Normalize to 0-1
                features['is_weekend'] = float(dt.weekday() >= 5)
                features['is_night'] = float(dt.hour < 6 or dt.hour > 22)
            else:
                # Default values if parsing fails
                features.update({
                    'hour_of_day': 0.5, 'day_of_week': 0.5, 
                    'is_weekend': 0.0, 'is_night': 0.0
                })
        except:
            features.update({
                'hour_of_day': 0.5, 'day_of_week': 0.5, 
                'is_weekend': 0.0, 'is_night': 0.0
            })
            
        return features
    
    def transform(self, emails_df) -> SparseMatrix:
        """Transform email metadata to feature vectors."""
        vectors = []
        
        for _, row in emails_df.iterrows():
            features = {}
            
            # Sender features
            if 'sender' in row and 'receiver' in row:
                features.update(self._extract_sender_features(row['sender'], row['receiver']))
            
            # Temporal features  
            if 'date' in row:
                features.update(self._extract_temporal_features(row['date']))
            
            # Convert to list format
            feature_names = sorted(features.keys())
            indices = list(range(len(feature_names)))
            values = [features[name] for name in feature_names]
            
            vector = SparseVector(indices, values, len(feature_names))
            vectors.append(vector)
        
        return SparseMatrix(vectors)


class Vectorizer:
    """Combined vectorizer that can use BoW, char n-grams, and heuristic features."""
    
    def __init__(self, config: FeatureConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.bow_vectorizer = None
        self.char_vectorizer = None
        self.heuristic_vectorizer = None
        self.total_dim = 0
        self._compute_dimensions()
    
    def _compute_dimensions(self):
        """Compute total dimensionality."""
        dim = 0
        if self.config['bow']:
            dim += self.config['hash_dim']
        if self.config['char_ngrams']:
            dim += self.config['hash_dim']
        if self.config['heuristics']:
            dim += 16  # Fixed number of heuristic features
        self.total_dim = dim
    
    def fit(self, texts: List[str]) -> 'Vectorizer':
        """Fit vectorizers on training data."""
        if self.config['bow']:
            self.bow_vectorizer = BagOfWordsVectorizer(
                hash_dim=self.config['hash_dim'], seed=self.seed
            ).fit(texts, use_tfidf=self.config.get('use_tfidf', True))
        
        if self.config['char_ngrams']:
            self.char_vectorizer = CharNGramVectorizer(
                ngram_min=self.config['ngram_min'],
                ngram_max=self.config['ngram_max'],
                hash_dim=self.config['hash_dim'],
                seed=self.seed
            )
        
        if self.config['heuristics']:
            self.heuristic_vectorizer = HeuristicVectorizer()
        
        return self
    
    def transform(self, texts: List[str]) -> SparseMatrix:
        """Transform texts to combined feature vectors."""
        combined_vectors = []
        
        # Get individual feature matrices
        matrices = []
        if self.config['bow'] and self.bow_vectorizer:
            matrices.append(self.bow_vectorizer.transform(texts))
        if self.config['char_ngrams'] and self.char_vectorizer:
            matrices.append(self.char_vectorizer.transform(texts))
        if self.config['heuristics'] and self.heuristic_vectorizer:
            matrices.append(self.heuristic_vectorizer.transform(texts))
        
        # Combine features
        for i in range(len(texts)):
            combined_indices = []
            combined_values = []
            offset = 0
            
            for matrix in matrices:
                vector = matrix[i]
                # Adjust indices by offset
                for idx, val in zip(vector.indices, vector.values):
                    combined_indices.append(idx + offset)
                    combined_values.append(val)
                offset += vector.dim
            
            # Create combined vector
            if combined_indices:
                # Sort by indices
                sorted_pairs = sorted(zip(combined_indices, combined_values))
                indices, values = zip(*sorted_pairs)
                vector = SparseVector(list(indices), list(values), self.total_dim)
            else:
                vector = SparseVector([], [], self.total_dim)
            
            # L2 normalize
            vector.normalize_l2_inplace()
            combined_vectors.append(vector)
        
        return SparseMatrix(combined_vectors)
    
    def save(self, path: str) -> None:
        """Save vectorizer configuration."""
        import json
        import os
        
        save_data = {
            'config': self.config,
            'seed': self.seed,
            'total_dim': self.total_dim
        }
        
        # Save IDF weights if using BoW with TF-IDF
        if self.bow_vectorizer and self.bow_vectorizer.use_tfidf:
            save_data['idf_weights'] = self.bow_vectorizer.idf_weights
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> 'Vectorizer':
        """Load vectorizer from file."""
        import json
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        vectorizer = Vectorizer(save_data['config'], save_data['seed'])
        
        # Restore IDF weights if present
        if 'idf_weights' in save_data and vectorizer.config['bow']:
            vectorizer.bow_vectorizer = BagOfWordsVectorizer(
                hash_dim=vectorizer.config['hash_dim'], 
                seed=vectorizer.seed
            )
            vectorizer.bow_vectorizer.use_tfidf = True
            # Convert string keys back to int
            vectorizer.bow_vectorizer.idf_weights = {
                int(k): v for k, v in save_data['idf_weights'].items()
            }
        
        if vectorizer.config['char_ngrams']:
            vectorizer.char_vectorizer = CharNGramVectorizer(
                ngram_min=vectorizer.config['ngram_min'],
                ngram_max=vectorizer.config['ngram_max'],
                hash_dim=vectorizer.config['hash_dim'],
                seed=vectorizer.seed
            )
        
        if vectorizer.config['heuristics']:
            vectorizer.heuristic_vectorizer = HeuristicVectorizer()
        
        return vectorizer