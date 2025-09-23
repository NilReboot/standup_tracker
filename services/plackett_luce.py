import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from pathlib import Path


class PlackettLuceModel(BaseEstimator, ClassifierMixin):
    """
    Custom Plackett-Luce model for ranking predictions using scikit-learn patterns.

    This model learns to predict the probability of selecting each candidate from a pool
    based on features, following the Plackett-Luce choice model framework.
    """

    def __init__(self, regularization_strength: float = 1.0, max_iter: int = 1000,
                 feature_scaling: bool = True, random_state: int = 42):
        """
        Initialize the Plackett-Luce model.

        Args:
            regularization_strength: L2 regularization strength for logistic regression
            max_iter: Maximum iterations for optimization
            feature_scaling: Whether to scale features before training
            random_state: Random seed for reproducibility
        """
        self.regularization_strength = regularization_strength
        self.max_iter = max_iter
        self.feature_scaling = feature_scaling
        self.random_state = random_state

        self.logistic_model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False

    def _prepare_features(self, features: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numpy array."""
        if not features:
            return np.array([]).reshape(0, 0)

        # Extract numeric features only (skip person_id)
        feature_keys = [k for k in features[0].keys() if k != 'person_id']

        if self.feature_names is None:
            self.feature_names = feature_keys

        X = np.array([[f[key] for key in self.feature_names] for f in features])
        return X

    def _create_training_pairs(self, ranking_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pairwise training data from ranking information.

        Args:
            ranking_data: List of dicts with keys: 'features' (list of feature dicts),
                         'chosen_person_id' (int), 'meeting_context' (dict)

        Returns:
            X: Feature differences between chosen and non-chosen candidates
            y: Binary labels (1 for chosen > non-chosen, 0 otherwise)
        """
        X_pairs = []
        y_pairs = []

        for ranking_instance in ranking_data:
            features = ranking_instance['features']
            chosen_id = ranking_instance['chosen_person_id']

            if not features or chosen_id is None:
                continue

            # Find the chosen candidate's features
            chosen_features = None
            other_features = []

            for f in features:
                if f['person_id'] == chosen_id:
                    chosen_features = f
                else:
                    other_features.append(f)

            if chosen_features is None or not other_features:
                continue

            # Create pairwise comparisons: chosen vs each non-chosen
            chosen_vector = np.array([chosen_features[key] for key in self.feature_names])

            for other_f in other_features:
                other_vector = np.array([other_f[key] for key in self.feature_names])

                # Feature difference: chosen - other (positive means chosen is better)
                feature_diff = chosen_vector - other_vector

                X_pairs.append(feature_diff)
                y_pairs.append(1)  # Chosen was preferred

                # Also add the reverse comparison for better training
                X_pairs.append(-feature_diff)
                y_pairs.append(0)  # Other was not preferred

        return np.array(X_pairs), np.array(y_pairs)

    def fit(self, ranking_data: List[Dict]) -> 'PlackettLuceModel':
        """
        Fit the Plackett-Luce model on ranking data.

        Args:
            ranking_data: List of ranking instances, each containing:
                - 'features': List of feature dicts for candidates
                - 'chosen_person_id': ID of the person who was actually chosen
                - 'meeting_context': Additional context (optional)

        Returns:
            self: Fitted model
        """
        if not ranking_data:
            raise ValueError("No training data provided")

        # Initialize feature names from first valid instance
        for instance in ranking_data:
            if instance['features']:
                X_sample = self._prepare_features(instance['features'])
                if X_sample.shape[0] > 0:
                    break

        # Create pairwise training data
        X_pairs, y_pairs = self._create_training_pairs(ranking_data)

        if len(X_pairs) == 0:
            raise ValueError("No valid pairwise comparisons could be created from training data")

        # Scale features if requested
        if self.feature_scaling:
            self.scaler = StandardScaler()
            X_pairs = self.scaler.fit_transform(X_pairs)

        # Train logistic regression on pairwise preferences
        self.logistic_model = LogisticRegression(
            C=1.0/self.regularization_strength,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs'
        )

        self.logistic_model.fit(X_pairs, y_pairs)
        self.is_fitted = True

        return self

    def _compute_plackett_luce_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert scores to Plackett-Luce choice probabilities.

        Args:
            scores: Array of candidate scores

        Returns:
            Array of selection probabilities (sum to 1)
        """
        # Avoid numerical overflow by subtracting max score
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)

        # Plackett-Luce probability: exp(score_i) / sum(exp(score_j))
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities

    def predict_proba(self, features: List[Dict]) -> np.ndarray:
        """
        Predict selection probabilities for candidates.

        Args:
            features: List of feature dictionaries for candidates

        Returns:
            Array of selection probabilities, one per candidate
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if not features:
            return np.array([])

        # Prepare features
        X = self._prepare_features(features)

        if X.shape[0] == 0:
            return np.array([])

        # Scale features if scaler was fitted
        if self.scaler is not None:
            # For prediction, we need to compute relative scores
            # We'll use the first candidate as reference and compute pairwise scores
            reference_features = X[0:1]  # Shape (1, n_features)

            scores = np.zeros(len(X))
            scores[0] = 0.0  # Reference candidate gets score 0

            for i in range(1, len(X)):
                # Compute feature difference: candidate_i - reference
                feature_diff = X[i:i+1] - reference_features

                if self.scaler is not None:
                    feature_diff = self.scaler.transform(feature_diff)

                # Predict probability that candidate_i is preferred over reference
                pref_prob = self.logistic_model.predict_proba(feature_diff)[0, 1]

                # Convert to relative score (logit of preference probability)
                if pref_prob >= 0.999:
                    pref_prob = 0.999
                elif pref_prob <= 0.001:
                    pref_prob = 0.001

                scores[i] = np.log(pref_prob / (1 - pref_prob))
        else:
            # Without scaler, compute scores similarly
            scores = np.zeros(len(X))
            reference_features = X[0:1]

            for i in range(1, len(X)):
                feature_diff = X[i:i+1] - reference_features
                pref_prob = self.logistic_model.predict_proba(feature_diff)[0, 1]

                if pref_prob >= 0.999:
                    pref_prob = 0.999
                elif pref_prob <= 0.001:
                    pref_prob = 0.001

                scores[i] = np.log(pref_prob / (1 - pref_prob))

        # Convert scores to Plackett-Luce probabilities
        probabilities = self._compute_plackett_luce_probabilities(scores)

        return probabilities

    def predict(self, features: List[Dict]) -> int:
        """
        Predict the most likely candidate to be selected.

        Args:
            features: List of feature dictionaries for candidates

        Returns:
            Index of the most likely candidate
        """
        probabilities = self.predict_proba(features)

        if len(probabilities) == 0:
            return 0

        return np.argmax(probabilities)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.logistic_model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_scores = np.abs(self.logistic_model.coef_[0])

        return dict(zip(self.feature_names, importance_scores))

    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'logistic_model': self.logistic_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hyperparams': {
                'regularization_strength': self.regularization_strength,
                'max_iter': self.max_iter,
                'feature_scaling': self.feature_scaling,
                'random_state': self.random_state
            }
        }

        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> 'PlackettLuceModel':
        """Load a fitted model from disk."""
        model_data = joblib.load(filepath)

        self.logistic_model = model_data['logistic_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        # Load hyperparameters
        hyperparams = model_data['hyperparams']
        self.regularization_strength = hyperparams['regularization_strength']
        self.max_iter = hyperparams['max_iter']
        self.feature_scaling = hyperparams['feature_scaling']
        self.random_state = hyperparams['random_state']

        self.is_fitted = True

        return self


def get_model_path() -> Path:
    """Get the standard path for saving/loading the model."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir / "plackett_luce_model.joblib"