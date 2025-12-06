"""
Complete Medical Image Classification System - Addressing ALL PDF Requirements
==============================================================================

Research Gap: Existing methods lack comprehensive uncertainty quantification
and multi-scale feature fusion with attention mechanisms.

Novel Solution: Attention-Enhanced Ensemble CNN with:
- Multi-backbone ensemble (ResNet50, EfficientNetB0, DenseNet121)
- Dual attention mechanism (channel + spatial)
- Uncertainty quantification (entropy + disagreement)
- Cross-validation and comparative analysis

Author: Research Implementation
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
import time
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


class AttentionBlock(layers.Layer):
    """Channel and Spatial Attention - Novel contribution"""
    def __init__(self, channels, reduction_ratio=16):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.fc1 = layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        self.conv_spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))
        channel_att = avg_out + max_out
        x = inputs * tf.expand_dims(tf.expand_dims(channel_att, 1), 1)
        
        # Spatial attention
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_att = self.conv_spatial(spatial_concat)
        
        return x * spatial_att


class BaselineCNN:
    """Simple baseline for comparison"""
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class MedicalImageClassifier:
    """Main classifier implementing novel attention-enhanced ensemble"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        self.history = []
        
    def build_model(self, backbone_name):
        """Build model with attention mechanism"""
        if backbone_name == 'resnet':
            base = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif backbone_name == 'efficientnet':
            base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            base = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        base.trainable = False
        
        inputs = layers.Input(shape=self.input_shape)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = base(x)
        x = AttentionBlock(channels=x.shape[-1])(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs, name=f'{backbone_name}_model')
    
    def create_ensemble(self):
        """Create 3-model ensemble"""
        print("\n" + "="*60)
        print("BUILDING ENSEMBLE ARCHITECTURE")
        print("="*60)
        for backbone in ['resnet', 'efficientnet', 'densenet']:
            print(f"Building {backbone.upper()}...")
            self.models.append(self.build_model(backbone))
    
    def compile_models(self):
        """Compile all models"""
        for model in self.models:
            model.compile(
                optimizer=keras.optimizers.Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            )
    
    def generate_data(self, n=1000, balanced=True):
        """Generate synthetic data (replace with real dataset)"""
        print(f"\nGenerating {'balanced' if balanced else 'imbalanced'} dataset...")
        X = np.random.rand(n, *self.input_shape).astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min())
        
        if balanced:
            y = np.repeat(range(self.num_classes), n // self.num_classes)
            if len(y) < n:
                y = np.concatenate([y, np.random.choice(self.num_classes, n - len(y))])
        else:
            y = np.concatenate([np.zeros(int(n*0.7)), np.ones(int(n*0.2)), 
                               np.full(n - int(n*0.9), 2)])
            np.random.shuffle(y)
        
        return X, keras.utils.to_categorical(y.astype(int), self.num_classes)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=15):
        """Train ensemble"""
        self.create_ensemble()
        self.compile_models()
        
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        for idx, model in enumerate(self.models):
            print(f"\n{'='*60}\nTraining Model {idx+1}/3: {model.name}\n{'='*60}")
            hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=epochs, batch_size=32, callbacks=callbacks_list, verbose=1)
            self.history.append(hist)
    
    def predict_ensemble(self, X):
        """Ensemble prediction with uncertainty"""
        predictions = [m.predict(X, verbose=0) for m in self.models]
        ensemble_pred = np.mean(predictions, axis=0)
        uncertainty = -np.sum(ensemble_pred * np.log(ensemble_pred + 1e-10), axis=1)
        disagreement = np.std(predictions, axis=0).mean(axis=1)
        return ensemble_pred, uncertainty, disagreement
    
    def cross_validate(self, X, y, n_splits=5):
        """K-fold cross-validation"""
        print(f"\n{'='*60}\n{n_splits}-FOLD CROSS-VALIDATION\n{'='*60}")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_labels = np.argmax(y, axis=1)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labels), 1):
            print(f"\nFold {fold}/{n_splits}")
            model = self.build_model('resnet')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]),
                     epochs=10, batch_size=32, verbose=0,
                     callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
            
            y_pred = np.argmax(model.predict(X[val_idx], verbose=0), axis=1)
            acc = accuracy_score(np.argmax(y[val_idx], axis=1), y_pred)
            scores.append(acc)
            print(f"Accuracy: {acc:.4f}")
        
        print(f"\nCV Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores
    
    def evaluate(self, X_test, y_test, baseline=None):
        """Comprehensive evaluation"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Ensemble
        print("\n[ENSEMBLE]")
        start = time.time()
        y_pred_prob, uncertainty, disagreement = self.predict_ensemble(X_test)
        infer_time = time.time() - start
        
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Uncertainty: {np.mean(uncertainty):.4f}")
        print(f"Inference: {infer_time:.2f}s ({len(X_test)/infer_time:.1f} samples/s)")
        
        results = {'ensemble': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                               'y_true': y_true, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob,
                               'confusion_matrix': confusion_matrix(y_true, y_pred)}}
        
        # Baseline comparison
        if baseline:
            print("\n[BASELINE]")
            y_pred_base = np.argmax(baseline.model.predict(X_test, verbose=0), axis=1)
            base_acc = accuracy_score(y_true, y_pred_base)
            base_prec, base_rec, base_f1, _ = precision_recall_fscore_support(y_true, y_pred_base, average='weighted', zero_division=0)
            
            print(f"Accuracy:  {base_acc:.4f}")
            print(f"Precision: {base_prec:.4f}")
            print(f"Recall:    {base_rec:.4f}")
            print(f"F1-Score:  {base_f1:.4f}")
            
            print(f"\n[IMPROVEMENT]")
            print(f"Accuracy:  +{(acc-base_acc)*100:.2f}%")
            print(f"F1-Score:  +{(f1-base_f1)*100:.2f}%")
            
            results['baseline'] = {'accuracy': base_acc, 'precision': base_prec, 
                                  'recall': base_rec, 'f1': base_f1}
        
        print("\n" + classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(self.num_classes)]))
        return results
    
    def visualize(self, results):
        """Generate visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Evaluation Results', fontsize=16, fontweight='bold')
        
        ens = results['ensemble']
        
        # Confusion Matrix
        sns.heatmap(ens['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True')
        axes[0,0].set_xlabel('Predicted')
        
        # ROC Curves
        y_true_bin = keras.utils.to_categorical(ens['y_true'], self.num_classes)
        for i in range(self.num_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], ens['y_pred_prob'][:, i])
                auc = roc_auc_score(y_true_bin[:, i], ens['y_pred_prob'][:, i])
                axes[0,1].plot(fpr, tpr, label=f'Class {i} (AUC={auc:.3f})', linewidth=2)
        axes[0,1].plot([0,1], [0,1], 'k--', label='Random')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(alpha=0.3)
        
        # Metrics Comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        ens_vals = [ens['accuracy'], ens['precision'], ens['recall'], ens['f1']]
        x = np.arange(len(metrics))
        axes[0,2].bar(x, ens_vals, color='#2ca02c', alpha=0.8, label='Ensemble')
        if 'baseline' in results:
            base = results['baseline']
            base_vals = [base['accuracy'], base['precision'], base['recall'], base['f1']]
            axes[0,2].bar(x + 0.3, base_vals, width=0.3, color='#ff7f0e', alpha=0.8, label='Baseline')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(metrics, rotation=15)
        axes[0,2].set_ylim([0, 1.1])
        axes[0,2].set_title('Performance Comparison')
        axes[0,2].legend()
        axes[0,2].grid(axis='y', alpha=0.3)
        
        # Confidence Distribution
        max_probs = np.max(ens['y_pred_prob'], axis=1)
        axes[1,0].hist(max_probs, bins=30, edgecolor='black', alpha=0.7)
        axes[1,0].axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2)
        axes[1,0].set_title('Prediction Confidence')
        axes[1,0].set_xlabel('Max Probability')
        axes[1,0].grid(alpha=0.3)
        
        # Training History
        if self.history:
            for idx, hist in enumerate(self.history):
                epochs = range(1, len(hist.history['accuracy']) + 1)
                axes[1,1].plot(epochs, hist.history['val_accuracy'], label=f'Model {idx+1}')
            axes[1,1].set_title('Validation Accuracy')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].legend()
            axes[1,1].grid(alpha=0.3)
        
        # Per-Class F1 Scores
        report = classification_report(ens['y_true'], ens['y_pred'], output_dict=True, zero_division=0)
        classes = [f'Class {i}' for i in range(self.num_classes)]
        f1_scores = [report[f'{i}']['f1-score'] for i in range(self.num_classes)]
        axes[1,2].bar(classes, f1_scores, color='#1f77b4', alpha=0.8)
        axes[1,2].set_title('Per-Class F1 Scores')
        axes[1,2].set_ylim([0, 1.1])
        axes[1,2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n[SAVED] results/comprehensive_evaluation.png")
        plt.show()


def main():
    """Main execution pipeline"""
    print("\n" + "="*60)
    print("MEDICAL IMAGE CLASSIFICATION RESEARCH")
    print("Attention-Enhanced Ensemble CNN")
    print("="*60)
    
    os.makedirs('results', exist_ok=True)
    
    # Initialize
    classifier = MedicalImageClassifier(input_shape=(224, 224, 3), num_classes=3)
    
    # Generate data
    print("\n[1] DATA GENERATION")
    X, y = classifier.generate_data(n=1000, balanced=True)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Cross-validation
    print("\n[2] CROSS-VALIDATION")
    cv_scores = classifier.cross_validate(X_train, y_train, n_splits=5)
    
    # Train ensemble
    print("\n[3] TRAINING ENSEMBLE")
    classifier.train(X_train, y_train, X_val, y_val, epochs=10)
    
    # Train baseline
    print("\n[4] TRAINING BASELINE")
    baseline = BaselineCNN(input_shape=(224, 224, 3), num_classes=3)
    baseline.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=10, batch_size=32, verbose=1,
                      callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    
    # Evaluate
    print("\n[5] EVALUATION")
    results = classifier.evaluate(X_test, y_test, baseline=baseline)
    
    # Visualize
    print("\n[6] VISUALIZATION")
    classifier.visualize(results)
    
    print("\n" + "="*60)
    print("RESEARCH COMPLETE")
    print("Check 'results/' directory for outputs")
    print("="*60)


if __name__ == '__main__':
    main()
