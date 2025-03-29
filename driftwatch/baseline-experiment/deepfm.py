from sklearn.preprocessing import LabelEncoder
from dataloader import get_device, DataLoader
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import torch
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driftwatch.embedding_tracker import EmbeddingTracker
from analyze_drift_metrics import DriftMetricsAnalyzer
from utils import *

class DeepFMModel:
    def __init__(self, data):
        self.data = data
        self.device = get_device()
        self.model = None
        self.feature_columns = None
        self.feature_names = None
        self.embedding_tracker = None
        self.embedding_dim = 8  # Matches embedding_dim used in sparse features
        # Store label encoders for each feature
        self.label_encoders = {}
        
    def prepare_features(self):
        sparse_features = ["parent_asin", "user_id", "year", "month", "day"]
        for feat in sparse_features:
            lbe = LabelEncoder()
            self.data.loc[:, feat] = lbe.fit_transform(self.data[feat])
            # Store the fitted encoder
            self.label_encoders[feat] = lbe
            
        fixlen_feature_columns = [
            SparseFeat("parent_asin", self.data["parent_asin"].nunique(), embedding_dim=8),
            SparseFeat("user_id", self.data["user_id"].nunique(), embedding_dim=8),
            SparseFeat("year", self.data["year"].nunique(), embedding_dim=8),
            SparseFeat("month", self.data["month"].nunique(), embedding_dim=8),
            SparseFeat("day", self.data["day"].nunique(), embedding_dim=8)
        ]
        return fixlen_feature_columns

    def extract_embeddings(self, inputs):
        """Extract embeddings from the DeepFM model's input."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Get sparse feature embeddings
        embeddings = []
        for name in self.feature_columns:
            feature_name = name.name
            embedding_table = self.model.embedding_dict[feature_name]
            # Get the feature's indices from the input
            indices = inputs[feature_name].long()
            # Extract the corresponding embeddings
            feature_embedding = embedding_table(indices)
            embeddings.append(feature_embedding)
        
        # Concatenate all embeddings and flatten
        combined_embedding = torch.cat(embeddings, dim=1)
        # Average over batch dimension to get a single embedding vector per instance
        return combined_embedding.mean(dim=0).detach().cpu().numpy()

    def train_model(self):
        print("\nDefining and training DeepFM model...")
        self.feature_columns = self.prepare_features()
        self.feature_names = get_feature_names(self.feature_columns + self.feature_columns)
        
        # Explicitly convert all features to int32 to prevent object_ arrays
        model_input = {}
        for name in self.feature_names:
            # Ensure all input data is int32 type
            model_input[name] = self.data[name].astype(np.int32).values
        
        # Initialize embedding tracker with the same dimension as our embeddings
        self.embedding_tracker = EmbeddingTracker(
            embedding_dim=len(self.feature_columns) * self.embedding_dim, 
            distance_name="euclidean"  # Can be changed to mahalanobis, manhattan, etc.
        )
        
        self.model = DeepFM(self.feature_columns, self.feature_columns, task='binary', device=self.device)
        self.model.compile("adam", loss='binary_crossentropy', metrics=["AUC"])
        
        # Train the model
        self.model.fit(
            model_input,
            self.data["rating"].values.astype(np.float32),  # Also ensure target is float32
            batch_size=1024,
            epochs=6,
            verbose=2,
            validation_split=0.1
        )
        
        # After training, establish baseline distribution by extracting embeddings
        with torch.no_grad():
            # Extract and update embedding tracker with training data
            batch_size = 1024
            num_samples = len(self.data)
            for i in range(0, num_samples, batch_size):
                batch_input = {name: torch.tensor(model_input[name][i:i+batch_size]).to(self.device) 
                               for name in self.feature_names}
                batch_embeddings = self.extract_embeddings(batch_input)
                self.embedding_tracker.update(batch_embeddings)
        
        print("DeepFM model training complete and baseline embeddings tracked.")

    def predict_with_drift_detection(self, inference_data):
        """Make predictions and check for drift in the inference data."""
        if self.model is None:
            raise ValueError("Model must be trained before inference")
            
        # Use stored label encoders from training for consistency
        inference_copy = inference_data.copy()
        for feat, encoder in self.label_encoders.items():
            # Handle unseen categories gracefully 
            # First create a mapping for all known values from training
            known_categories = {val: idx for idx, val in enumerate(encoder.classes_)}
            
            # For each value in inference data, map it if known, or use a default value
            inference_copy.loc[:, feat] = inference_copy[feat].map(
                lambda x: known_categories.get(x, len(known_categories))  # Use next available index for new values
            )
            
        # Explicitly convert to int32 type
        model_input = {}
        for name in self.feature_names:
            model_input[name] = inference_copy[name].astype(np.int32).values
            
        # Extract embeddings from inference data
        with torch.no_grad():
            inference_input = {name: torch.tensor(model_input[name]).to(self.device) 
                              for name in self.feature_names}
            inference_embeddings = self.extract_embeddings(inference_input)
            
        # Check for drift using the embedding tracker
        drift_score = self.embedding_tracker.compute_distance(inference_embeddings)
        
        # Make predictions
        predictions = self.model.predict(model_input, batch_size=1024)
        
        return predictions, drift_score

    def save_model(self, model_path, embedding_tracker_path=None):
        """Save the trained model and embedding tracker to files."""
        # Save the actual DeepFM model
        torch.save(self.model.state_dict(), model_path)
        
        # If path provided, save the embedding tracker which contains baseline distribution
        if embedding_tracker_path and self.embedding_tracker:
            with open(embedding_tracker_path, 'wb') as f:
                pickle.dump(self.embedding_tracker, f)
                
        # Save label encoders
        encoder_path = model_path + ".encoders"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        # Save feature information
        metadata = {
            'feature_columns': self.feature_columns,
            'feature_names': self.feature_names,
            'embedding_dim': self.embedding_dim
        }
        metadata_path = model_path + ".metadata"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path, embedding_tracker_path=None):
        """Load a pre-trained model and embedding tracker."""
        # Load metadata first to set up model structure
        metadata_path = model_path + ".metadata"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.feature_columns = metadata['feature_columns']
        self.feature_names = metadata['feature_names']
        self.embedding_dim = metadata['embedding_dim']
        
        # Load label encoders
        encoder_path = model_path + ".encoders"
        with open(encoder_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Initialize model with correct structure
        self.model = DeepFM(self.feature_columns, self.feature_columns, task='binary', device=self.device)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        
        # Load embedding tracker if path provided
        if embedding_tracker_path and os.path.exists(embedding_tracker_path):
            with open(embedding_tracker_path, 'rb') as f:
                self.embedding_tracker = pickle.load(f)
                print(f"Loaded embedding tracker from {embedding_tracker_path}")
        else:
            print("Warning: No embedding tracker loaded. Drift detection will not work without baseline distribution.")
        
        print(f"Model loaded from {model_path}")
        return self



def main():
    data_loader = DataLoader("Amazon_Fashion.jsonl")
    data = data_loader.load_and_preprocess_data()
    
    # Split data into training and test sets (simulating baseline and inference data)
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Check if we have a pre-trained model
    model_path = "../models/deepfm_model.pt"
    tracker_path = "../models/embedding_tracker.pkl"
    
    if os.path.exists(model_path) and os.path.exists(tracker_path):
        print("Loading pre-trained model...")
        model = DeepFMModel(None)  # Initialize with None as no training data needed
        model.load_model(model_path, tracker_path)
    else:
        print("Training new model...")
        model = DeepFMModel(train_data)
        model.train_model()
        
        # Save the model for future use
        os.makedirs("../models", exist_ok=True)
        model.save_model(model_path, tracker_path)
    
    # Create an instance of the DriftMetricsAnalyzer
    analyzer = DriftMetricsAnalyzer(results_folder="../drift_results")
    
    # Test all metrics from metrics.py
    vector_metrics = ["euclidean", "manhattan", "minkowski", "chebyshev", "canberra", "mahalanobis"]
    distribution_metrics = ["kl", "js", "hellinger", "bhattacharyya", "wasserstein", "mmd"]
    
    # Combine all metrics to test
    metrics_to_test = vector_metrics + distribution_metrics
    
    # Use a smaller set of drift strengths to keep the experiment manageable
    drift_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for metric in metrics_to_test:
        print(f"\nTesting drift detection with {metric} metric:")
        
        # Create a new model with the specific metric
        model_metric = DeepFMModel(train_data)
        model_metric.embedding_tracker = EmbeddingTracker(
            embedding_dim=len(model_metric.prepare_features()) * model_metric.embedding_dim, 
            distance_name=metric
        )
        model_metric.train_model()
        
        # Measure drift at different strengths
        drift_scores = []
        avg_predictions = []
        
        for strength in drift_strengths:
            # Apply controlled drift based on strength
            drifted_data = apply_controlled_drift(test_data, strength)
            
            # Measure drift and make predictions
            predictions, drift_score = model_metric.predict_with_drift_detection(drifted_data)
            
            drift_scores.append(drift_score)
            avg_predictions.append(predictions.mean())
            
            print(f"  Drift strength {strength}: score = {drift_score:.4f}, avg prediction = {predictions.mean():.4f}")
        
        # Save the results for this metric
        analyzer.save_experiment_results(metric, drift_strengths, drift_scores, avg_predictions)
    
    # Generate comparative analysis plot
    analyzer.plot_comparative_analysis()
    print("\nDrift metrics analysis complete. Check the 'drift_results' folder for the generated visualizations.")

if __name__ == "__main__":
    main()
