"""
Model registry for managing predictive models in the Financial Risk Analysis System
"""
import logging
import os
import pickle
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import importlib
import datetime

from src.core.config import settings


class Model:
    """Base class for predictive models"""
    
    def __init__(self, name: str, model_type: str, version: str = "1.0.0", 
                 description: str = None, trained_date: datetime.datetime = None):
        """Initialize a model
        
        Args:
            name: Name of the model
            model_type: Type of model (e.g., 'classification', 'regression', 'time_series')
            version: Version string
            description: Description of what the model does
            trained_date: When the model was trained
        """
        self.name = name
        self.model_type = model_type
        self.version = version
        self.description = description
        self.trained_date = trained_date or datetime.datetime.now()
        self.model = None
        self.metadata = {}
        
    def predict(self, data):
        """Make predictions using the model
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return self.model.predict(data)
    
    def get_feature_importance(self):
        """Get feature importance from the model if available
        
        Returns:
            Feature importance or None if not available
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return self.model.coef_
        else:
            return None
    
    def save(self, directory: Path = settings.MODEL_SAVE_DIR):
        """Save the model to disk
        
        Args:
            directory: Directory to save the model in
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        directory.mkdir(exist_ok=True, parents=True)
        
        # Create a filename based on model name and version
        filename = f"{self.name.lower().replace(' ', '_')}_{self.version.replace('.', '_')}.pkl"
        file_path = directory / filename
        
        # Save the model using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata separately
        metadata_path = directory / f"{self.name.lower().replace(' ', '_')}_{self.version.replace('.', '_')}_metadata.pkl"
        metadata = {
            'name': self.name,
            'model_type': self.model_type,
            'version': self.version,
            'description': self.description,
            'trained_date': self.trained_date,
            'metadata': self.metadata
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return file_path
    
    def load(self, file_path: Path):
        """Load the model from disk
        
        Args:
            file_path: Path to the saved model file
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata if available
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.metadata = metadata.get('metadata', {})
        
        return self


class ModelRegistry:
    """Registry for all predictive models"""
    
    def __init__(self):
        """Initialize the model registry"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ModelRegistry")
        
        # Initialize model registry
        self.models: Dict[str, Model] = {}
        
        # Load existing models if any
        if settings.USE_ML_MODELS:
            self._load_existing_models()
        
        self.logger.info(f"ModelRegistry initialized with {len(self.models)} models")
    
    def register_model(self, model: Model):
        """Register a model in the registry
        
        Args:
            model: The model to register
        """
        self.models[model.name] = model
        self.logger.info(f"Registered model: {model.name} (v{model.version})")
    
    def get_model(self, model_name: str) -> Optional[Model]:
        """Get a model by name
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            The model, or None if not found
        """
        return self.models.get(model_name)
    
    def get_all_models(self) -> List[Model]:
        """Get all registered models
        
        Returns:
            List of all models
        """
        return list(self.models.values())
    
    def _load_existing_models(self):
        """Load existing models from the model directory"""
        model_dir = settings.MODEL_SAVE_DIR
        
        if not model_dir.exists():
            self.logger.info(f"Model directory {model_dir} does not exist. No models loaded.")
            return
        
        # Find all pickle files that don't have '_metadata' in the name
        model_files = [f for f in model_dir.glob("*.pkl") if "_metadata" not in f.name]
        
        for model_file in model_files:
            try:
                # Extract model name and version from filename
                name_version = model_file.stem
                parts = name_version.split('_')
                if len(parts) > 1:
                    version_parts = parts[-2:]
                    name_parts = parts[:-2]
                    version = '.'.join(version_parts)
                    name = '_'.join(name_parts)
                else:
                    name = name_version
                    version = "1.0.0"
                
                model = Model(name=name, model_type="unknown", version=version)
                model.load(model_file)
                self.register_model(model)
                
                self.logger.info(f"Loaded model: {model.name} (v{model.version})")
                
            except Exception as e:
                self.logger.error(f"Error loading model from {model_file}: {e}")
    
    def save_all_models(self):
        """Save all registered models to disk"""
        for model_name, model in self.models.items():
            try:
                file_path = model.save()
                self.logger.info(f"Saved model: {model_name} to {file_path}")
            except Exception as e:
                self.logger.error(f"Error saving model {model_name}: {e}")
    
    def train_market_risk_model(self, data):
        """Train a market risk prediction model
        
        Args:
            data: Training data for the model
        
        Returns:
            Trained model
        """
        self.logger.info("Training market risk model")
        
        try:
            # This would typically use scikit-learn, PyTorch, or other ML libraries
            # For demonstration, we'll just create a placeholder model
            from sklearn.ensemble import RandomForestRegressor
            
            X = data['features']
            y = data['target']
            
            model_obj = RandomForestRegressor(n_estimators=100, random_state=42)
            model_obj.fit(X, y)
            
            # Create model wrapper
            model = Model(
                name="MarketRiskModel",
                model_type="regression",
                version="1.0.0",
                description="Predicts market risk factors using random forest regression"
            )
            model.model = model_obj
            model.metadata = {
                'feature_names': data.get('feature_names', []),
                'target_name': data.get('target_name', 'unknown'),
                'training_date': datetime.datetime.now().isoformat(),
                'n_samples': len(X)
            }
            
            # Register the model
            self.register_model(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training market risk model: {e}")
            raise
    
    def train_credit_risk_model(self, data):
        """Train a credit risk prediction model
        
        Args:
            data: Training data for the model
        
        Returns:
            Trained model
        """
        self.logger.info("Training credit risk model")
        # Placeholder for credit risk model training
        return None
    
    def train_systemic_risk_model(self, data):
        """Train a systemic risk prediction model
        
        Args:
            data: Training data for the model
        
        Returns:
            Trained model
        """
        self.logger.info("Training systemic risk model")
        # Placeholder for systemic risk model training
        return None
    
    def get_top_features(self, model_name: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N important features for a model
        
        Args:
            model_name: Name of the model
            n: Number of top features to return
        
        Returns:
            List of feature importance dictionaries
        """
        model = self.get_model(model_name)
        if not model:
            self.logger.warning(f"Model {model_name} not found")
            return []
        
        feature_importance = model.get_feature_importance()
        if feature_importance is None:
            self.logger.warning(f"No feature importance available for model {model_name}")
            return []
        
        # Get feature names from metadata if available
        feature_names = model.metadata.get('feature_names', [])
        if not feature_names or len(feature_names) != len(feature_importance):
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create list of feature importance dictionaries
        features = [
            {'name': name, 'importance': float(importance)}
            for name, importance in zip(feature_names, feature_importance)
        ]
        
        # Sort by importance and take top N
        features.sort(key=lambda x: x['importance'], reverse=True)
        return features[:n] 