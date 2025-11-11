# ============================================================================
# PADS Dataset DataLoader for Hierarchical Cross-Attention Model
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.cross_attention_2to1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        self.self_attention_1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        
        # Layer norms for residual connections
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        # Cross attention with residual connections
        channel_1_cross_attn, _ = self.cross_attention_1to2(query=channel_1,key=channel_2,value=channel_2)
        channel_1_cross = self.norm_cross_1(channel_1 + channel_1_cross_attn)
        
        channel_2_cross_attn, _ = self.cross_attention_2to1(query=channel_2,key=channel_1,value=channel_1)
        channel_2_cross = self.norm_cross_2(channel_2 + channel_2_cross_attn)
        
        # Self attention with residual connections
        channel_1_self_attn, _ = self.self_attention_1(query=channel_1_cross,key=channel_1_cross,value=channel_1_cross)
        channel_1_self = self.norm_self_1(channel_1_cross + channel_1_self_attn)
        
        channel_2_self_attn, _ = self.self_attention_2(query=channel_2_cross,key=channel_2_cross,value=channel_2_cross)
        channel_2_self = self.norm_self_2(channel_2_cross + channel_2_self_attn)
        
        # Feed forward
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out

class MyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_window_layers: int = 4,  # Number of cross-attention layers at window level
        num_task_layers: int = 2,    # Number of attention layers at task level
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 256,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.seq_len = seq_len

        # ========== LEVEL 1: Window-Level Processing ==========
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, max_len=seq_len)

        # Cross-attention layers
        self.window_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(num_window_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ========== LEVEL 2: Task-Level Processing ==========
        self.window_positional_encoding = PositionalEncoding(model_dim * 2, max_len=100)  # max 100 windows

        self.task_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=model_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm': nn.LayerNorm(model_dim * 2),
                'feed_forward': FeedForward(model_dim * 2, d_ff, dropout)
            })
            for _ in range(num_task_layers)
        ])

        # Task-level pooling
        self.task_attention_pooling = nn.Sequential(
            nn.Linear(model_dim * 2, 1),
            nn.Softmax(dim=1)
        )

        # ========== Classification Heads ==========
        # No text encoder - using only signal features
        fusion_dim = model_dim * 2

        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: HC vs PD
        )

        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: PD vs DD
        )

        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, batch):
       
        device = batch['left_windows'].device
        batch_size = batch['left_windows'].shape[0]
        max_windows = batch['left_windows'].shape[1]
        
        # ========== LEVEL 1: Window-Level Cross-Attention ==========
        # Reshape: (batch, max_windows, 256, 6) -> (batch * max_windows, 256, 6)
        left_windows_flat = batch['left_windows'].view(-1, self.seq_len, 6)
        right_windows_flat = batch['right_windows'].view(-1, self.seq_len, 6)
        
        # Project to model dimension
        left_encoded = self.left_projection(left_windows_flat)   # (batch*max_windows, 256, model_dim)
        right_encoded = self.right_projection(right_windows_flat) 
        
        # Add positional encoding
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        # Apply cross-attention layers between left and right wrist
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)
        
        # Global pooling for each window
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)  # (batch*max_windows, model_dim)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1) # (batch*max_windows, model_dim)
        
        # Concatenate left and right features
        window_features = torch.cat([left_pool, right_pool], dim=1)  # (batch*max_windows, model_dim*2)
        
        # ========== LEVEL 2: Task-Level Attention ==========
        # Reshape back to (batch, max_windows, model_dim*2)
        window_features = window_features.view(batch_size, max_windows, -1)
        
        window_features = self.window_positional_encoding(window_features)
        
        # Create attention mask for padding (invert the mask: True -> False for valid positions)
        key_padding_mask = ~batch['masks']  # (batch_size, max_windows)
        
        # Apply task-level self-attention layers
        task_features = window_features
        for task_layer in self.task_layers:
            attn_output, _ = task_layer['self_attention'](
                query=task_features,
                key=task_features,
                value=task_features,
                key_padding_mask=key_padding_mask
            )
            task_features = task_layer['norm'](task_features + attn_output)
            task_features = task_layer['feed_forward'](task_features)
        
        # Attention-based pooling to get task representation
        attention_weights = self.task_attention_pooling(task_features)  # (batch, max_windows, 1)
        
        attention_weights = attention_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        task_representation = (task_features * attention_weights).sum(dim=1)  # (batch, model_dim*2)

        # ========== Classification ==========
        # Using only signal features (no text)
        logits_hc_vs_pd = self.head_hc_vs_pd(task_representation)
        logits_pd_vs_dd = self.head_pd_vs_dd(task_representation)
        
        return logits_hc_vs_pd, logits_pd_vs_dd
    
    
    def get_features(self, batch):
        """Extract features for tsne plot"""
        device = batch['left_windows'].device
        batch_size = batch['left_windows'].shape[0]
        max_windows = batch['left_windows'].shape[1]
        
        # Level 1: Window processing
        left_windows_flat = batch['left_windows'].view(-1, self.seq_len, 6)
        right_windows_flat = batch['right_windows'].view(-1, self.seq_len, 6)
        
        left_encoded = self.left_projection(left_windows_flat)
        right_encoded = self.right_projection(right_windows_flat)
        
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)
        
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)
        
        window_features = torch.cat([left_pool, right_pool], dim=1)
        window_features = window_features.view(batch_size, max_windows, -1)
        
        # Level 2: Task processing
        window_features = self.window_positional_encoding(window_features)
        key_padding_mask = ~batch['masks']
        
        task_features = window_features
        for task_layer in self.task_layers:
            attn_output, _ = task_layer['self_attention'](
                query=task_features,
                key=task_features,
                value=task_features,
                key_padding_mask=key_padding_mask
            )
            task_features = task_layer['norm'](task_features + attn_output)
            task_features = task_layer['feed_forward'](task_features)
        
        attention_weights = self.task_attention_pooling(task_features)
        attention_weights = attention_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        task_representation = (task_features * attention_weights).sum(dim=1)

        return {
            'window_features': window_features,  # (batch, max_windows, model_dim*2)
            'task_representation': task_representation,  # (batch, model_dim*2)
            'fused_features': task_representation,  # (batch, model_dim*2) - no text, just signal features
            'attention_weights': attention_weights.squeeze(-1)  # (batch, max_windows)
        }
# ============================================================================
# Evaluation functions
# ============================================================================
def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'confusion_matrix': cm
    }
    
    if verbose and task_name:
        print(f"\n=== {task_name} Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision_avg:.4f}")
        print(f" Recall: {recall_avg:.4f}")
        print(f"F1: {f1_avg:.4f}")
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_name = "HC" if label == 0 else ("PD" if label == 1 else f"Class_{label}")
                if task_name == "PD vs DD":
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics

def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd):

    os.makedirs("metrics", exist_ok=True)

    # helper writer
    def write_csv(filename, metrics_list):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
            for epoch_data in metrics_list:
                writer.writerow([
                    epoch_data['epoch'],
                    epoch_data['metrics'].get('accuracy', 0),
                    epoch_data['metrics'].get('precision', 0),
                    epoch_data['metrics'].get('recall', 0),
                    epoch_data['metrics'].get('f1', 0)
                ])

    # HC vs PD
    if fold_metrics_hc:
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")

    # PD vs DD
    if fold_metrics_pd:
        pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")



def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir="plots"):
    
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    valid_hc_pd = hc_pd_labels != -1
    valid_pd_dd = pd_dd_labels != -1

    # plot HC vs PD
    if np.any(valid_hc_pd):
        plt.figure(figsize=(8, 6))
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        pd_mask = labels_hc_pd == 1
        
        if np.any(hc_mask):
            plt.scatter(features_hc_pd[hc_mask,0], features_hc_pd[hc_mask,1], 
                        c='blue', label=f'HC (n={np.sum(hc_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask,0], features_hc_pd[pd_mask,1], 
                        c='red', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: HC vs PD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_hc_vs_pd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[saved] tsne_hc_vs_pd.png")

    # plot PD vs DD
    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]

        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask,0], features_pd_dd[pd_mask,1], 
                        c='green', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask,0], features_pd_dd[dd_mask,1], 
                        c='orange', label=f'DD (n={np.sum(dd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: PD vs DD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[saved] tsne_pd_vs_dd.png")

    return features_2d


# ============================================================================
# Helper functions 
# ============================================================================
def create_windows(data, window_size=256, overlap=0):
    
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))   
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])
    
    return np.array(windows) if windows else None


def downsample(data, original_freq=100, target_freq=64):
    
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
   
    nyquist = 0.5 * original_freq
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


class PADSDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        subject_ids: List[str],
        window_size: int = 256,
        stride: int = 128,
        classification_task: str = 'hc_vs_pd',
        max_windows_per_task: int = 20,
        skip_initial_samples: int = 50,  # Skip first 0.5s to remove vibration
        tasks_to_include: Optional[List[str]] = None,
        apply_downsampling: bool = True,
        apply_filtering: bool = True,
        original_freq: int = 100,
        target_freq: int = 64,
        use_differential_sampling: bool = True  # For class imbalance handling
    ):
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.stride = stride
        self.classification_task = classification_task
        self.max_windows_per_task = max_windows_per_task
        self.skip_initial_samples = skip_initial_samples
        self.apply_downsampling = apply_downsampling
        self.apply_filtering = apply_filtering
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.use_differential_sampling = use_differential_sampling
        
        # Default tasks - all 11 movement tasks (some split into 2 parts)
        self.all_tasks = [
            "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", 
            "LiftArms", "StretchHold", "HoldWeight", "PointingRight",
            "PointingLeft", "DrinkGlas", "CrossArms", "TouchNose",
            "Entrainment1", "Entrainment2"
        ]
        
        self.tasks_to_include = tasks_to_include if tasks_to_include else self.all_tasks
        
        # Load patient metadata
        self.patient_data = self._load_patient_data()
        
        # Filter subjects based on classification task
        self.valid_subjects = self._filter_subjects_by_task()
        
        print(f"Loaded {len(self.valid_subjects)} subjects for {classification_task}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_patient_data(self) -> Dict:
        """Load patient metadata from JSON files."""
        patient_data = {}
        for subject_id in self.subject_ids:
            patient_file = self.data_dir / "patients" / f"patient_{subject_id}.json"
            if patient_file.exists():
                with open(patient_file, 'r') as f:
                    data = json.load(f)
                    patient_data[subject_id] = data
        return patient_data
    
    def _filter_subjects_by_task(self) -> List[str]:
        """Filter subjects based on classification task."""
        valid_subjects = []
        
        for subject_id in self.subject_ids:
            if subject_id not in self.patient_data:
                continue
            
            condition = self.patient_data[subject_id]['condition']
            
            if self.classification_task == 'hc_vs_pd':
                # Include Healthy Controls and PD patients
                if condition in ['Healthy', 'PD', 'Parkinsons Disease', 'Parkinson']:
                    valid_subjects.append(subject_id)
            
            elif self.classification_task == 'pd_vs_dd':
                # Include PD patients and Differential Diagnoses
                if condition not in ['Healthy']:
                    valid_subjects.append(subject_id)
        
        return valid_subjects
    
    def _get_class_distribution(self) -> Dict:
        """Get distribution of classes in the dataset."""
        distribution = {}
        for subject_id in self.valid_subjects:
            condition = self.patient_data[subject_id]['condition']
            distribution[condition] = distribution.get(condition, 0) + 1
        return distribution
    
    def _get_label(self, subject_id: str) -> Tuple[int, int]:
        condition = self.patient_data[subject_id]['condition']
        
        # HC vs PD label
        if condition == 'Healthy':
            hc_vs_pd_label = 0
        else:
            hc_vs_pd_label = 1
        
        # PD vs DD label
        if condition in ['PD', 'Parkinsons Disease', 'Parkinson']:
            pd_vs_dd_label = 0
        else:
            pd_vs_dd_label = 1
        
        return hc_vs_pd_label, pd_vs_dd_label
    
    def _load_movement_data(self, subject_id: str) -> Dict[str, np.ndarray]:
        observation_file = self.data_dir / "movement" / f"observation_{subject_id}.json"
        
        if not observation_file.exists():
            return {}
        
        with open(observation_file, 'r') as f:
            observation_data = json.load(f)
        
        movement_data = {}
        
        for session in observation_data.get('session', []):
            task_name = session['record_name']
            
            # Load data for each wrist
            for record in session['records']:
                device_location = record['device_location']
                
                # Read the time series file
                file_path = self.data_dir / "movement" / "timeseries" / record['file_name'].replace('bins/', '').replace('.bin', '.txt')
                
                if file_path.exists():
                    # Load CSV data
                    data = np.loadtxt(file_path, delimiter=',')
                    
                    # Key format: TaskName_Wrist (e.g., "Relaxed_Left")
                    key = f"{task_name}_{device_location}"
                    movement_data[key] = data
        
        return movement_data
    
    def _get_overlap_for_condition(self, condition: str) -> float:
        if not self.use_differential_sampling:
            return 0.0
        
        if condition == 'Healthy':
            return 0.7  # More overlap = more windows for healthy controls
        elif 'Parkinson' in condition or condition == 'PD':
            return 0.0  # No overlap for PD patients
        else:
            return 0.65  # High overlap for differential diagnoses
    
    def _create_windows_with_overlap(self, data: np.ndarray, overlap: float) -> np.ndarray:
        # Skip initial samples to remove vibration artifact
        data = data[self.skip_initial_samples:]
        
        # Apply preprocessing
        if self.apply_downsampling:
            data = downsample(data, self.original_freq, self.target_freq)
        
        if self.apply_filtering:
            data = bandpass_filter(data, self.target_freq if self.apply_downsampling else self.original_freq)
        
        # Create windows using helper function
        windows = create_windows(data, self.window_size, overlap)
        
        return windows if windows is not None else np.array([])
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        data = data[self.skip_initial_samples:]

        n_samples = len(data)
        if n_samples < self.window_size:
            return np.array([])

        windows = []
        for start_idx in range(0, n_samples - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = data[start_idx:end_idx]
            windows.append(window)

        return np.array(windows)
    
    def __len__(self) -> int:
        return len(self.valid_subjects)
    
    def __getitem__(self, idx: int) -> Dict:
       
        subject_id = self.valid_subjects[idx]
        
        # Load movement data
        movement_data = self._load_movement_data(subject_id)
        
        # Collect windows from all tasks
        all_left_windows = []
        all_right_windows = []
        
        for task in self.tasks_to_include:
            left_key = f"{task}_LeftWrist"
            right_key = f"{task}_RightWrist"
            
            if left_key in movement_data and right_key in movement_data:
                left_data = movement_data[left_key]
                right_data = movement_data[right_key]
                
                # Create windows
                left_windows = self._create_windows(left_data)
                right_windows = self._create_windows(right_data)
                
                # Limit number of windows per task
                if len(left_windows) > 0 and len(right_windows) > 0:
                    n_windows = min(len(left_windows), len(right_windows), self.max_windows_per_task)
                    all_left_windows.append(left_windows[:n_windows])
                    all_right_windows.append(right_windows[:n_windows])
        
        # Concatenate all windows
        if len(all_left_windows) > 0:
            left_windows = np.concatenate(all_left_windows, axis=0)
            right_windows = np.concatenate(all_right_windows, axis=0)
        else:
            # Handle case with no valid windows
            left_windows = np.zeros((1, self.window_size, 6))
            right_windows = np.zeros((1, self.window_size, 6))
        
        # Get number of actual windows
        n_windows = len(left_windows)
        
        # Get labels
        label_hc_vs_pd, label_pd_vs_dd = self._get_label(subject_id)
        
        return {
            'left_windows': left_windows,
            'right_windows': right_windows,
            'n_windows': n_windows,
            'label_hc_vs_pd': label_hc_vs_pd,
            'label_pd_vs_dd': label_pd_vs_dd,
            'subject_id': subject_id
        }


def collate_fn(batch: List[Dict]) -> Dict:
   
    # Find max number of windows in batch
    max_windows = max([item['n_windows'] for item in batch])
    
    batch_size = len(batch)
    window_size = batch[0]['left_windows'].shape[1]
    
    # Initialize tensors
    left_windows = torch.zeros(batch_size, max_windows, window_size, 6)
    right_windows = torch.zeros(batch_size, max_windows, window_size, 6)
    masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    labels_hc_vs_pd = torch.zeros(batch_size, dtype=torch.long)
    labels_pd_vs_dd = torch.zeros(batch_size, dtype=torch.long)
    subject_ids = []
    
    for i, item in enumerate(batch):
        n_windows = item['n_windows']
        
        # Copy actual windows
        left_windows[i, :n_windows] = torch.from_numpy(item['left_windows']).float()
        right_windows[i, :n_windows] = torch.from_numpy(item['right_windows']).float()
        
        # Set mask for valid windows
        masks[i, :n_windows] = True
        
        # Copy labels
        labels_hc_vs_pd[i] = item['label_hc_vs_pd']
        labels_pd_vs_dd[i] = item['label_pd_vs_dd']
        
        subject_ids.append(item['subject_id'])
    
    return {
        'left_windows': left_windows,
        'right_windows': right_windows,
        'masks': masks,
        'label_hc_vs_pd': labels_hc_vs_pd,
        'label_pd_vs_dd': labels_pd_vs_dd,
        'subject_ids': subject_ids
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    window_size: int = 256,
    stride: int = 128,
    classification_task: str = 'hc_vs_pd',
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
   
    # Get all subject IDs
    patient_dir = Path(data_dir) / "patients"
    all_subject_ids = []
    
    for patient_file in sorted(patient_dir.glob("patient_*.json")):
        subject_id = patient_file.stem.split('_')[1]
        all_subject_ids.append(subject_id)
    
    # Shuffle and split
    np.random.seed(random_seed)
    np.random.shuffle(all_subject_ids)
    
    n_total = len(all_subject_ids)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_ids = all_subject_ids[:n_train]
    val_ids = all_subject_ids[n_train:n_train + n_val]
    test_ids = all_subject_ids[n_train + n_val:]
    
    print(f"Dataset split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Create datasets
    train_dataset = PADSDataset(
        data_dir=data_dir,
        subject_ids=train_ids,
        window_size=window_size,
        stride=stride,
        classification_task=classification_task
    )
    
    val_dataset = PADSDataset(
        data_dir=data_dir,
        subject_ids=val_ids,
        window_size=window_size,
        stride=stride,
        classification_task=classification_task
    )
    
    test_dataset = PADSDataset(
        data_dir=data_dir,
        subject_ids=test_ids,
        window_size=window_size,
        stride=stride,
        classification_task=classification_task
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Create dataloaders
    data_dir = "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0"
    classification_task = 'hc_vs_pd'
    num_epochs = 10

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        window_size=256,
        stride=128,
        classification_task=classification_task,
        num_workers=4
    )
    
    # Test loading a batch
    for batch in train_loader:
        print("Batch shapes:")
        print(f"  Left windows: {batch['left_windows'].shape}")
        print(f"  Right windows: {batch['right_windows'].shape}")
        print(f"  Masks: {batch['masks'].shape}")
        print(f"  Labels (HC vs PD): {batch['label_hc_vs_pd']}")
        print(f"  Labels (PD vs DD): {batch['label_pd_vs_dd']}")
        print(f"  Subject IDs: {batch['subject_ids']}")
        break
    
    # Example: Training loop integration

    model = MyModel(
        input_dim=6,
        model_dim=128,
        num_heads=8,
        num_window_layers=4,
        num_task_layers=2,
        d_ff=512,
        dropout=0.1,
        seq_len=256
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Track training losses
    train_losses = []

    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Classification task: {classification_task}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            logits_hc_vs_pd, logits_pd_vs_dd = model(batch)

            # Compute loss (choose based on classification_task)
            if classification_task == 'hc_vs_pd':
                loss = criterion(logits_hc_vs_pd, batch['label_hc_vs_pd'])
            else:
                loss = criterion(logits_pd_vs_dd, batch['label_pd_vs_dd'])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"\n>>> Epoch [{epoch+1}/{num_epochs}] Complete - Average Loss: {avg_loss:.4f}\n")

    # Plot training loss curve
    print(f"\n{'='*60}")
    print("Training Complete! Generating loss plot...")
    print(f"{'='*60}\n")

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Loss', fontsize=12, fontweight='bold')
    plt.title(f'Training Loss Curve - {classification_task.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    loss_plot_path = f"plots/training_loss_{classification_task}.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Loss plot saved: {loss_plot_path}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Best training loss: {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses)) + 1})")