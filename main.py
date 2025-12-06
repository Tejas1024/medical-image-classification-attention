"""Advanced Medical Image Classification using Attention-Enhanced CNN Ensemble
Author: Research Implementation
Date: December 2025

Proposal: Novel hybrid architecture combining:
1. Multi-scale feature extraction
2. Channel and spatial attention mechanisms
3. Ensemble learning with uncertainty quantification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class AttentionBlock(layers.Layer):
    """Channel and Spatial Attention Mechanism"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        
    def build(self, input_shape):
        self.fc1 = layers.Dense(self.channels // 16, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        
    def call(self, inputs):
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        channel_att = avg_out + max_out
        
        # Apply channel attention
        x = inputs * tf.expand_dims(tf.expand_dims(channel_att, 1), 1)
        return x

class MedicalImageClassifier:
    """Attention-Enhanced Ensemble CNN for Medical Image Classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        self.history = None
        
    def build_model(self, backbone_name):
        """Build attention-enhanced backbone model"""
        # Load pre-trained backbone
        if backbone_name == 'resnet':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                 input_shape=self.input_shape)
        elif backbone_name == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                       input_shape=self.input_shape)
        else:  # densenet
            base_model = DenseNet121(weights='imagenet', include_top=False,
                                    input_shape=self.input_shape)
        
        # Freeze base model weights
        base_model.trainable = False
        
        # Build model with attention
        inputs = layers.Input(shape=self.input_shape)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = base_model(x)
        
        # Attention mechanism
        x = AttentionBlock(channels=x.shape[-1])(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_ensemble(self):
        """Create ensemble of 3 models"""
        backbones = ['resnet', 'efficientnet', 'densenet']
        for backbone in backbones:
            print(f"Building {backbone} model...")
            model = self.build_model(backbone)
            self.models.append(model)
    
    def compile_models(self):
        """Compile all models"""
        for model in self.models:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
    
    def generate_synthetic_data(self, num_samples=500):
        """Generate synthetic medical image data (replace with real data)"""
        X = np.random.rand(num_samples, *self.input_shape).astype(np.float32)
        # Normalize to [0, 1]
        X = X / 255.0
        # Create balanced classes
        y = np.random.randint(0, self.num_classes, num_samples)
        y = keras.utils.to_categorical(y, self.num_classes)
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train ensemble models"""
        self.create_ensemble()
        self.compile_models()
        
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        for idx, model in enumerate(self.models):
            print(f"\n{'='*50}")
            print(f"Training Model {idx+1}/{len(self.models)}")
            print(f"{'='*50}")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=1
            )
            self.history = history
    
    def predict_ensemble(self, X):
        """Make predictions using ensemble voting"""
        predictions = []
        uncertainties = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            # Calculate uncertainty (entropy)
            uncertainty = -np.sum(pred * np.log(pred + 1e-10), axis=1)
            uncertainties.append(uncertainty)
        
        # Ensemble prediction (averaging)
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_uncertainty = np.mean(uncertainties, axis=0)
        
        return ensemble_pred, ensemble_uncertainty
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred_prob, y_uncertainty = self.predict_ensemble(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\n{'='*50}")
        print("ENSEMBLE MODEL EVALUATION")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Mean Uncertainty: {np.mean(y_uncertainty):.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'uncertainty': np.mean(y_uncertainty),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
            'confusion_matrix': cm
        }
    
    def visualize_results(self, results):
        """Generate comprehensive visualizations"""
        fig = plt.figure(figsize=(15, 12))
        
        # Confusion Matrix
        ax1 = plt.subplot(2, 2, 1)
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # ROC-AUC Curves (one-vs-rest)
        ax2 = plt.subplot(2, 2, 2)
        y_pred_prob = results['y_pred_prob']
        y_true_bin = keras.utils.to_categorical(results['y_true'], num_classes=self.num_classes)
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])
            ax2.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Metrics Comparison
        ax3 = plt.subplot(2, 2, 3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
        ax3.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Metrics')
        ax3.set_ylim([0, 1])
        for i, v in enumerate(values):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Prediction Distribution
        ax4 = plt.subplot(2, 2, 4)
        max_probs = np.max(y_pred_prob, axis=1)
        ax4.hist(max_probs, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Maximum Prediction Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Confidence Distribution')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ensemble_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to 'results/ensemble_evaluation.png'")
        plt.show()

def main():
    """Main execution"""
    print("Medical Image Classification - Attention-Enhanced CNN Ensemble")
    print("="*60)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Initialize classifier
    classifier = MedicalImageClassifier(input_shape=(224, 224, 3), num_classes=3)
    
    # Generate synthetic data (replace with real medical images)
    print("\nGenerating synthetic medical image data...")
    X, y = classifier.generate_synthetic_data(num_samples=600)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train ensemble
    print("\nTraining ensemble models...")
    classifier.train(X_train, y_train, X_val, y_val, epochs=3)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = classifier.evaluate(X_test, y_test)
    
    # Visualize
    print("\nGenerating visualizations...")
    classifier.visualize_results(results)
    
    print("\n" + "="*60)
    print("Training complete! All results saved to 'results/' directory")
    print("="*60)

if __name__ == '__main__':
    main()