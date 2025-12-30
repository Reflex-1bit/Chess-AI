"""
ML Model Training for Chess Mistake Prediction

Trains models to predict:
1. Mistake probability (classification)
2. Expected eval loss (regression)

Uses Random Forest for interpretability and robustness.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import pickle
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ChessMistakeModel:
    """
    Trains and uses ML models to predict chess mistakes.
    
    Uses two models:
    1. Classifier: Predicts if a move is a mistake (binary)
    2. Regressor: Predicts expected eval loss (continuous)
    """
    
    def __init__(self, mistake_threshold: float = 100.0, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            mistake_threshold: Eval loss threshold (centipawns) to label as mistake
            random_state: Random seed for reproducibility
        """
        self.mistake_threshold = mistake_threshold
        self.random_state = random_state
        
        self.mistake_classifier = None
        self.eval_loss_regressor = None
        self.feature_names = None
        self.is_trained = False
    
    def prepare_data(self, moves_data: List[Dict], feature_extractor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from parsed moves.
        
        Args:
            moves_data: List of move dictionaries from PGNParser
            feature_extractor: FeatureExtractor instance
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y_mistake: Binary labels (1 = mistake, 0 = not mistake)
            y_eval_loss: Continuous labels (eval loss in centipawns)
        """
        X_list = []
        y_mistake_list = []
        y_eval_loss_list = []
        
        for move_data in moves_data:
            # Extract features from position before move
            board_before = move_data['board_before']
            is_white = move_data['is_white']
            
            features = feature_extractor.extract_features(board_before, is_white)
            X_list.append(features)
            
            # Labels
            eval_loss = move_data['eval_loss']
            y_eval_loss_list.append(eval_loss)
            
            # Binary mistake label
            is_mistake = 1 if eval_loss > self.mistake_threshold else 0
            y_mistake_list.append(is_mistake)
        
        X = np.array(X_list)
        y_mistake = np.array(y_mistake_list)
        y_eval_loss = np.array(y_eval_loss_list)
        
        # Store feature names
        self.feature_names = feature_extractor.get_feature_names()
        
        return X, y_mistake, y_eval_loss
    
    def train(self, X: np.ndarray, y_mistake: np.ndarray, y_eval_loss: np.ndarray,
              test_size: float = 0.2, game_indices: List[int] = None) -> Dict:
        """
        Train both classification and regression models.
        
        Args:
            X: Feature matrix
            y_mistake: Binary mistake labels
            y_eval_loss: Continuous eval loss labels
            test_size: Fraction of data for testing
            game_indices: Optional list of game indices for each sample (for game-level splits)
        
        Returns:
            Dictionary with training metrics
        """
        # Split data
        if game_indices is not None:
            # Game-level split to avoid data leakage
            unique_games = list(set(game_indices))
            train_games, test_games = train_test_split(
                unique_games, test_size=test_size, random_state=self.random_state
            )
            train_mask = np.array([g in train_games for g in game_indices])
            test_mask = np.array([g in test_games for g in game_indices])
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_mistake_train, y_mistake_test = y_mistake[train_mask], y_mistake[test_mask]
            y_eval_loss_train, y_eval_loss_test = y_eval_loss[train_mask], y_eval_loss[test_mask]
        else:
            # Random split (less ideal but works if game indices unavailable)
            X_train, X_test, y_mistake_train, y_mistake_test, y_eval_loss_train, y_eval_loss_test = \
                train_test_split(X, y_mistake, y_eval_loss, test_size=test_size, random_state=self.random_state)
        
        # Train mistake classifier
        print("Training mistake classifier...")
        self.mistake_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,  # Limit depth to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.mistake_classifier.fit(X_train, y_mistake_train)
        
        # Train eval loss regressor
        print("Training eval loss regressor...")
        self.eval_loss_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.eval_loss_regressor.fit(X_train, y_eval_loss_train)
        
        self.is_trained = True
        
        # Evaluate
        metrics = self._evaluate(X_test, y_mistake_test, y_eval_loss_test)
        return metrics
    
    def _evaluate(self, X_test: np.ndarray, y_mistake_test: np.ndarray, 
                  y_eval_loss_test: np.ndarray) -> Dict:
        """Evaluate models on test set."""
        # Classification metrics
        y_mistake_pred = self.mistake_classifier.predict(X_test)
        y_mistake_proba = self.mistake_classifier.predict_proba(X_test)[:, 1]
        
        # Regression metrics
        y_eval_loss_pred = self.eval_loss_regressor.predict(X_test)
        
        metrics = {
            'classification_report': classification_report(y_mistake_test, y_mistake_pred),
            'mistake_accuracy': np.mean(y_mistake_pred == y_mistake_test),
            'mistake_precision': np.sum((y_mistake_pred == 1) & (y_mistake_test == 1)) / max(np.sum(y_mistake_pred == 1), 1),
            'mistake_recall': np.sum((y_mistake_pred == 1) & (y_mistake_test == 1)) / max(np.sum(y_mistake_test == 1), 1),
            'eval_loss_mse': mean_squared_error(y_eval_loss_test, y_eval_loss_pred),
            'eval_loss_rmse': np.sqrt(mean_squared_error(y_eval_loss_test, y_eval_loss_pred)),
            'eval_loss_r2': r2_score(y_eval_loss_test, y_eval_loss_pred),
            'feature_importance': self.get_feature_importance()
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mistake probability and expected eval loss.
        
        Args:
            X: Feature matrix
        
        Returns:
            mistake_proba: Probability of mistake (0-1)
            expected_eval_loss: Expected eval loss in centipawns
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        mistake_proba = self.mistake_classifier.predict_proba(X)[:, 1]
        expected_eval_loss = self.eval_loss_regressor.predict(X)
        
        return mistake_proba, expected_eval_loss
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from both models.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Average importance from both models
        classifier_importance = self.mistake_classifier.feature_importances_
        regressor_importance = self.eval_loss_regressor.feature_importances_
        avg_importance = (classifier_importance + regressor_importance) / 2
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance,
            'classifier_importance': classifier_importance,
            'regressor_importance': regressor_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'mistake_classifier': self.mistake_classifier,
            'eval_loss_regressor': self.eval_loss_regressor,
            'feature_names': self.feature_names,
            'mistake_threshold': self.mistake_threshold,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(mistake_threshold=model_data['mistake_threshold'])
        model.mistake_classifier = model_data['mistake_classifier']
        model.eval_loss_regressor = model_data['eval_loss_regressor']
        model.feature_names = model_data['feature_names']
        model.is_trained = model_data['is_trained']
        
        return model

