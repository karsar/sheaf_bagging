import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import time
import json
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

REAL_DATASET_IDENTIFIERS = [
    'breast-cancer', 'diabetes', 'credit-g', 'SPECTF', 'spambase', 'ionosphere', 'sonar', 'ilpd', 'tic-tac-toe', 'qsar-biodeg', 
    'haberman', 'heart-statlog', 'kr-vs-kp',
    'ozone-level-8hr', 'phoneme',
]

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1_units = 16
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.fc1_units),
            nn.ReLU(),
            nn.Linear(self.fc1_units, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_net(net, X_train, y_train, n_epochs=500, lr=1e-3, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    if not isinstance(X_train, torch.Tensor):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
    else:
        X_tensor = X_train
    if not isinstance(y_train, torch.Tensor):
        y_tensor = torch.tensor(y_train, dtype=torch.long)
    else:
        y_tensor = y_train
    
    if X_tensor.shape[0] == 0:
        return net
    
    # Create DataLoader for minibatch training
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    net.train()
    for epoch in range(n_epochs):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return net

def load_openml_dataset(dataset_name):
    """Load and preprocess an OpenML dataset by name using sklearn."""
    try:
        # Fetch dataset by name using sklearn (removed parser parameter)
        data = fetch_openml(name=dataset_name, version=1, as_frame=True)
        X = data.data.copy()  # Make explicit copy
        y = data.target.copy()  # Make explicit copy
        
        print(f"Loaded raw dataset '{dataset_name}': {X.shape[0]} samples, {X.shape[1]} features")
        
        # Handle missing values in features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # Categorical: fill with mode
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val[0])
                else:
                    X[col] = X[col].fillna('unknown')
                # Label encode categorical features - convert to object first, then assign
                le = LabelEncoder()
                encoded_values = le.fit_transform(X[col].astype(str))
                X[col] = encoded_values.astype('float64')  # Convert to numeric dtype
            else:
                # Numerical: fill with median
                X[col] = X[col].fillna(X[col].median())
        
        # Convert to numpy array
        X = X.values.astype(np.float32)
        
        # Handle target variable missing values
        if y.dtype == 'object' or y.dtype.name == 'category':
            # Fill missing targets with mode
            mode_target = y.mode()
            if len(mode_target) > 0:
                y = y.fillna(mode_target[0])
            # Label encode target
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        else:
            # Numerical target: fill with median and ensure integer labels
            y = y.fillna(y.median())
            # Convert to integer labels if they're not already
            unique_vals = np.unique(y)
            if len(unique_vals) <= 20:  # Assume classification if <= 20 unique values
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
        
        # Remove any remaining NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Processed dataset '{dataset_name}': {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features, {len(np.unique(y))} classes")
        return X_scaled, y, dataset_name
        
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return None, None, None

def baseline_bagging(X_eval, y_eval, X_bagging_source, y_bagging_source, num_classes, n_models=5, n_epochs=500, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng_internal = np.random.default_rng()
    n_samples_bagging_source = len(X_bagging_source)
    base_model_preds_on_X_eval = []
    input_dim = X_bagging_source.shape[1]
    if n_samples_bagging_source == 0 or input_dim == 0:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    for _ in range(n_models):
        idx_base_model_train = rng_internal.choice(n_samples_bagging_source, size=n_samples_bagging_source, replace=True)
        X_base_train = X_bagging_source[idx_base_model_train]
        y_base_train = y_bagging_source[idx_base_model_train]
        if X_base_train.shape[0] == 0 or len(np.unique(y_base_train)) < min(2, num_classes):
            continue
        net_model = Net(input_dim, num_classes)
        net_model = train_net(net_model, X_base_train, y_base_train, n_epochs=n_epochs, batch_size=batch_size, device=device)
        net_model.eval()
        with torch.no_grad():
            X_tensor_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)
            logits_eval = net_model(X_tensor_eval)
            pred_base_model = torch.argmax(logits_eval, dim=1).cpu().numpy()
            base_model_preds_on_X_eval.append(pred_base_model)
    if not base_model_preds_on_X_eval:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    final_pred_on_X_eval = np.round(np.mean(base_model_preds_on_X_eval, axis=0)).astype(int)
    acc_on_X_eval = accuracy_score(y_eval, final_pred_on_X_eval)
    return final_pred_on_X_eval, acc_on_X_eval

def sheaf_bagging_random_proj_nn(X_eval, y_eval, X_training_data, y_training_data, num_classes,
                                 n_models=5, proj_dim=2, glue_weight=1.0, n_epochs=500, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples_train, d_train = X_training_data.shape
    if n_samples_train == 0 or d_train == 0 or proj_dim == 0:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    rng_internal = np.random.default_rng()
    projections_numpy = []
    nets = []
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_models):
        if d_train == proj_dim:
            Q, _ = np.linalg.qr(rng_internal.standard_normal((d_train, d_train)))
            R_np = Q
        else:
            R_np = rng_internal.standard_normal((d_train, proj_dim))
        projections_numpy.append(R_np)
        net_model = Net(proj_dim, num_classes).to(device)
        optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
        nets.append(net_model)
        optimizers.append(optimizer)
    X_train_tensor = torch.tensor(X_training_data, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_training_data, dtype=torch.long).to(device)
    if X_train_tensor.shape[0] == 0 or len(torch.unique(y_train_tensor)) < min(2, num_classes):
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    for epoch in range(n_epochs):
        for i in range(n_models):
            nets[i].train()
            R_i_torch = torch.tensor(projections_numpy[i], dtype=torch.float32).to(device)
            X_proj_i_train = X_train_tensor @ R_i_torch
            y_logits_i_train = nets[i](X_proj_i_train)
            loss_i = criterion(y_logits_i_train, y_train_tensor)
            glue_loss_val = torch.tensor(0.0, device=device)
            if n_models > 1 and glue_weight > 0:
                prob_i_train = nn.Softmax(dim=1)(y_logits_i_train)
                for j in range(n_models):
                    if i == j: continue
                    R_j_torch = torch.tensor(projections_numpy[j], dtype=torch.float32).to(device)
                    X_proj_j_train = X_train_tensor @ R_j_torch
                    original_mode_j = nets[j].training
                    nets[j].eval()
                    with torch.no_grad(): y_logits_j_train = nets[j](X_proj_j_train)
                    nets[j].train(mode=original_mode_j)
                    prob_j_train = nn.Softmax(dim=1)(y_logits_j_train)
                    glue_loss_val += torch.mean((prob_i_train - prob_j_train) ** 2)
            total_loss = loss_i + glue_weight * glue_loss_val
            optimizers[i].zero_grad()
            total_loss.backward()
            optimizers[i].step()
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    base_model_preds_on_X_eval = []
    for i in range(n_models):
        nets[i].eval()
        R_i_torch = torch.tensor(projections_numpy[i], dtype=torch.float32).to(device)
        X_proj_i_eval = X_eval_tensor @ R_i_torch
        with torch.no_grad():
            logits_eval = nets[i](X_proj_i_eval)
            pred_base_model = torch.argmax(logits_eval, dim=1).cpu().numpy()
            base_model_preds_on_X_eval.append(pred_base_model)
    final_pred_on_X_eval = np.round(np.mean(base_model_preds_on_X_eval, axis=0)).astype(int)
    acc_on_X_eval = accuracy_score(y_eval, final_pred_on_X_eval)
    return final_pred_on_X_eval, acc_on_X_eval

def sheaf_bagging_identity_proj_nn(X_eval, y_eval, X_training_data, y_training_data, num_classes,
                                   n_models=5, glue_weight=1.0, n_epochs=500, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples_train, d_train = X_training_data.shape
    if n_samples_train == 0 or d_train == 0:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    nets = []
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    for _ in range(n_models):
        net_model = Net(d_train, num_classes).to(device)
        optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
        nets.append(net_model)
        optimizers.append(optimizer)
    X_train_tensor = torch.tensor(X_training_data, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_training_data, dtype=torch.long).to(device)
    if X_train_tensor.shape[0] == 0 or len(torch.unique(y_train_tensor)) < min(2, num_classes):
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    for epoch in range(n_epochs):
        for i in range(n_models):
            nets[i].train()
            y_logits_i_train = nets[i](X_train_tensor)
            loss_i = criterion(y_logits_i_train, y_train_tensor)
            glue_loss_val = torch.tensor(0.0, device=device)
            if n_models > 1 and glue_weight > 0:
                prob_i_train = nn.Softmax(dim=1)(y_logits_i_train)
                for j in range(n_models):
                    if i == j: continue
                    original_mode_j = nets[j].training
                    nets[j].eval()
                    with torch.no_grad(): y_logits_j_train = nets[j](X_train_tensor)
                    nets[j].train(mode=original_mode_j)
                    prob_j_train = nn.Softmax(dim=1)(y_logits_j_train)
                    glue_loss_val += torch.mean((prob_i_train - prob_j_train) ** 2)
            total_loss = loss_i + glue_weight * glue_loss_val
            optimizers[i].zero_grad()
            total_loss.backward()
            optimizers[i].step()
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    base_model_preds_on_X_eval = []
    for i in range(n_models):
        nets[i].eval()
        with torch.no_grad():
            logits_eval = nets[i](X_eval_tensor)
            pred_base_model = torch.argmax(logits_eval, dim=1).cpu().numpy()
            base_model_preds_on_X_eval.append(pred_base_model)
    final_pred_on_X_eval = np.round(np.mean(base_model_preds_on_X_eval, axis=0)).astype(int)
    acc_on_X_eval = accuracy_score(y_eval, final_pred_on_X_eval)
    return final_pred_on_X_eval, acc_on_X_eval

def simple_ensemble_nn(X_eval, y_eval, X_training_data, y_training_data, num_classes,
                       n_models=5, n_epochs=500, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples_train, d_train = X_training_data.shape
    if n_samples_train == 0 or d_train == 0:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    base_model_preds_on_X_eval = []
    for _ in range(n_models):
        net_model = Net(d_train, num_classes).to(device)
        net_model = train_net(net_model, X_training_data, y_training_data, n_epochs=n_epochs, batch_size=batch_size, device=device)
        net_model.eval()
        with torch.no_grad():
            X_tensor_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)
            logits_eval = net_model(X_tensor_eval)
            pred_base_model = torch.argmax(logits_eval, dim=1).cpu().numpy()
            base_model_preds_on_X_eval.append(pred_base_model)
    if not base_model_preds_on_X_eval:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    final_pred_on_X_eval = np.round(np.mean(base_model_preds_on_X_eval, axis=0)).astype(int)
    acc_on_X_eval = accuracy_score(y_eval, final_pred_on_X_eval)
    return final_pred_on_X_eval, acc_on_X_eval

def prediction_disagreement(pred1, pred2):
    if len(pred1) != len(pred2) or len(pred1) == 0:
        return np.nan
    return np.mean(pred1 != pred2)

def calculate_stability_and_accuracy(method, X_eval, y_eval, X_train_source, y_train_source,
                                     input_dim_processed, num_classes,
                                     n_bootstrap_runs=5, device=None, **kwargs):
    preds_on_eval_list = []
    accuracies_on_eval = []
    bootstrap_rng_seed = kwargs.get('bootstrap_seed', None)
    bootstrap_rng = np.random.default_rng(seed=bootstrap_rng_seed)
    n_train_samples_source = len(X_train_source)
    if n_train_samples_source == 0 or input_dim_processed == 0:
        dummy_preds_for_all_runs = [np.array([])] * n_bootstrap_runs
        nan_acc_for_all_runs = [np.nan] * n_bootstrap_runs
        return np.nan, np.nan, [], np.nan, np.nan
    for bootstrap_iter in range(n_bootstrap_runs):
        idx_boot_train = bootstrap_rng.choice(n_train_samples_source, size=n_train_samples_source, replace=True)
        X_boot_for_training = X_train_source[idx_boot_train]
        y_boot_for_training = y_train_source[idx_boot_train]
        current_pred_on_X_eval = None
        current_acc_on_X_eval = np.nan
        n_epochs_arg = kwargs.get('n_epochs')
        n_models_arg = kwargs.get('n_models')
        glue_weight_arg = kwargs.get('glue_weight')
        proj_dim_for_net = kwargs.get('proj_dim')
        batch_size = kwargs.get('batch_size', 32)
        try:
            if X_boot_for_training.shape[0] == 0 or len(np.unique(y_boot_for_training)) < min(2, num_classes):
                preds_on_eval_list.append(np.array([]))
                accuracies_on_eval.append(np.nan)
                continue
            if method == 'standard':
                current_pred_on_X_eval, current_acc_on_X_eval = baseline_bagging(
                    X_eval, y_eval, X_boot_for_training, y_boot_for_training, num_classes,
                    n_models=n_models_arg, n_epochs=n_epochs_arg, device=device, batch_size=batch_size
                )
            elif method == 'sheaf_random_proj':
                current_pred_on_X_eval, current_acc_on_X_eval = sheaf_bagging_random_proj_nn(
                    X_eval, y_eval, X_boot_for_training, y_boot_for_training, num_classes,
                    n_models=n_models_arg, proj_dim=proj_dim_for_net,
                    glue_weight=glue_weight_arg, n_epochs=n_epochs_arg, device=device
                )
            elif method == 'sheaf_identity_proj':
                current_pred_on_X_eval, current_acc_on_X_eval = sheaf_bagging_identity_proj_nn(
                    X_eval, y_eval, X_boot_for_training, y_boot_for_training, num_classes,
                    n_models=n_models_arg, glue_weight=glue_weight_arg,
                    n_epochs=n_epochs_arg, device=device
                )
            elif method == 'single_nn':
                net = Net(input_dim_processed, num_classes)
                trained_net = train_net(net, X_boot_for_training, y_boot_for_training,
                                        n_epochs=n_epochs_arg, device=device)
                trained_net.eval()
                with torch.no_grad():
                    X_tensor_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)
                    logits_eval = trained_net(X_tensor_eval)
                    current_pred_on_X_eval = torch.argmax(logits_eval, dim=1).cpu().numpy()
                    current_acc_on_X_eval = accuracy_score(y_eval, current_pred_on_X_eval)
            elif method == 'simple_ensemble_nn':
                current_pred_on_X_eval, current_acc_on_X_eval = simple_ensemble_nn(
                    X_eval, y_eval, X_boot_for_training, y_boot_for_training, num_classes,
                    n_models=n_models_arg, n_epochs=n_epochs_arg, device=device, batch_size=batch_size
                )
            elif method == 'hybrid_sheaf_bagging':
                current_pred_on_X_eval, current_acc_on_X_eval = hybrid_sheaf_bagging_nn(
                    X_eval, y_eval, X_boot_for_training, y_boot_for_training, num_classes,
                    n_models=n_models_arg, glue_weight=glue_weight_arg,
                    glue_subsample_ratio=kwargs.get('glue_subsample_ratio', 1.0),
                    n_epochs=n_epochs_arg, device=device
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"Error in {method} for iter {bootstrap_iter}: {e}")
            current_pred_on_X_eval = None
            current_acc_on_X_eval = np.nan
        preds_on_eval_list.append(current_pred_on_X_eval if current_pred_on_X_eval is not None else np.array([]))
        accuracies_on_eval.append(current_acc_on_X_eval)
    valid_preds_list = [p for p in preds_on_eval_list if p is not None and p.ndim > 0 and p.shape[0] == len(y_eval)]
    disagreements = []
    if len(valid_preds_list) > 1:
        for i in range(len(valid_preds_list)):
            for j in range(i + 1, len(valid_preds_list)):
                dis = prediction_disagreement(valid_preds_list[i], valid_preds_list[j])
                if not np.isnan(dis): disagreements.append(dis)
    stability = np.mean(disagreements) if disagreements else np.nan
    stability_std = np.std(disagreements) if len(disagreements) > 1 else np.nan
    valid_accuracies = [acc for acc in accuracies_on_eval if not np.isnan(acc)]
    acc_mean = np.mean(valid_accuracies) if valid_accuracies else np.nan
    acc_std = np.std(valid_accuracies) if len(valid_accuracies) > 1 else np.nan
    
    return stability, acc_mean, disagreements, stability_std, acc_std

def hybrid_sheaf_bagging_nn(X_eval, y_eval, X_training_data, y_training_data, num_classes,
                             n_models=5, glue_weight=1.0, glue_subsample_ratio=1.0,
                             n_epochs=500, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples_train, d_train = X_training_data.shape
    if n_samples_train == 0 or d_train == 0:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    rng_internal_bagging = np.random.default_rng()
    rng_internal_glue_sampling = np.random.default_rng()
    nets = []
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    X_bag_tensors = []
    y_bag_tensors = []
    for _ in range(n_models):
        net_model = Net(d_train, num_classes).to(device)
        optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
        nets.append(net_model)
        optimizers.append(optimizer)
        idx_bag = rng_internal_bagging.choice(n_samples_train, size=n_samples_train, replace=True)
        X_b = X_training_data[idx_bag]
        y_b = y_training_data[idx_bag]
        X_bag_tensors.append(torch.tensor(X_b, dtype=torch.float32).to(device))
        y_bag_tensors.append(torch.tensor(y_b, dtype=torch.long).to(device))
    X_train_tensor_full = torch.tensor(X_training_data, dtype=torch.float32).to(device)
    for epoch in range(n_epochs):
        X_glue_tensor_current_epoch = None
        has_glue_data_current_epoch = False
        if n_samples_train > 0 and 0.0 < glue_subsample_ratio <= 1.0:
            n_glue_samples = int(n_samples_train * glue_subsample_ratio)
            if n_glue_samples == 0 and n_samples_train > 0: n_glue_samples = 1
            if n_glue_samples > 0:
                indices_np = np.arange(n_samples_train)
                idx_glue = rng_internal_glue_sampling.choice(indices_np, size=n_glue_samples, replace=(n_glue_samples > n_samples_train))
                idx_glue_torch = torch.tensor(idx_glue, dtype=torch.long).to(X_train_tensor_full.device)
                X_glue_tensor_current_epoch = X_train_tensor_full[idx_glue_torch]
                has_glue_data_current_epoch = X_glue_tensor_current_epoch.shape[0] > 0
        elif glue_subsample_ratio > 1.0:
            X_glue_tensor_current_epoch = X_train_tensor_full
            has_glue_data_current_epoch = X_glue_tensor_current_epoch.shape[0] > 0
        for i in range(n_models):
            nets[i].train()
            y_logits_pred_i = nets[i](X_bag_tensors[i])
            loss_pred_i = criterion(y_logits_pred_i, y_bag_tensors[i])
            glue_loss_val = torch.tensor(0.0, device=device)
            if n_models > 1 and glue_weight > 0 and has_glue_data_current_epoch:
                y_logits_glue_i = nets[i](X_glue_tensor_current_epoch)
                prob_i_glue = nn.Softmax(dim=1)(y_logits_glue_i)
                for j in range(n_models):
                    if i == j: continue
                    original_mode_j = nets[j].training
                    nets[j].eval()
                    with torch.no_grad():
                        y_logits_glue_j = nets[j](X_glue_tensor_current_epoch)
                    nets[j].train(mode=original_mode_j)
                    prob_j_glue = nn.Softmax(dim=1)(y_logits_glue_j)
                    glue_loss_val += torch.mean((prob_i_glue - prob_j_glue) ** 2)
            total_loss = loss_pred_i + glue_weight * glue_loss_val
            optimizers[i].zero_grad()
            total_loss.backward()
            optimizers[i].step()
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    base_model_preds_on_X_eval = []
    for i in range(n_models):
        nets[i].eval()
        with torch.no_grad():
            logits_eval = nets[i](X_eval_tensor)
            pred_base_model = torch.argmax(logits_eval, dim=1).cpu().numpy()
            base_model_preds_on_X_eval.append(pred_base_model)
    if not base_model_preds_on_X_eval:
        dummy_preds = np.zeros(len(y_eval), dtype=int) if len(y_eval) > 0 else np.array([], dtype=int)
        return dummy_preds, np.nan
    final_pred_on_X_eval = np.round(np.mean(base_model_preds_on_X_eval, axis=0)).astype(int)
    acc_on_X_eval = accuracy_score(y_eval, final_pred_on_X_eval)
    return final_pred_on_X_eval, acc_on_X_eval

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_models_ensemble_list = [2, 5, 10, 15]
    n_bootstrap_stability_runs = 20
    n_epochs_training = 500
    common_glue_weight = 0.2
    hybrid_glue_subsample_ratio = 0.5

    line_styles_config = {
        'single_nn': {'marker': 'x', 'color': 'green', 'linestyle': '--', 'label': 'Single NN'},
        'standard': {'marker': 'o', 'color': 'blue', 'label': 'Std. Bagging'},
        'sheaf_random_proj': {'marker': 's', 'color': 'orange', 'label': 'Sheaf RandProj'},
        'sheaf_identity_proj': {'marker': '^', 'color': 'purple', 'label': 'Sheaf IdentProj'},
        'hybrid_sheaf_bagging': {'marker': 'D', 'color': 'red', 'label': 'Hybrid Sheaf Bagging'},
        'simple_ensemble_nn': {'marker': 'P', 'color': 'brown', 'label': 'Simple Ensemble NN'}
    }

    all_dataset_method_diff_results = []
    accuracies_by_dataset = {}
    stabilities_by_dataset = {}

    # Loop over all datasets
    for dataset_name in REAL_DATASET_IDENTIFIERS:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        X_full, y_full, dataset_id_name = load_openml_dataset(dataset_name)
        
        if X_full is None:
            print(f"Skipping dataset {dataset_name} due to loading error")
            continue
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
        )

        print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        num_classes = len(np.unique(y_full))
        input_dim_processed = X_train.shape[1]  # Use train shape
        
        # Initialize data structures for this dataset
        accuracies_by_dataset[dataset_id_name] = {}
        stabilities_by_dataset[dataset_id_name] = {}

        # Initialize results structure with additional fields for std
        results_current_dataset = {name: {'stabilities': [], 'accuracies': [], 
                                         'stabilities_std': [], 'accuracies_std': [],
                                         'disagreements_hist_data': None}
                                   for name in line_styles_config.keys()}
        outer_bootstrap_seed = 123

        # --- Single NN ---
        print(f"\nEvaluating Single Neural Network for {dataset_id_name}...")
        start_time = time.time()
        s_single, a_single, d_list_single, s_single_std, a_single_std = calculate_stability_and_accuracy(
            'single_nn', X_test, y_test, X_train, y_train,  # Test for eval, train for training
            input_dim_processed, num_classes,
            n_bootstrap_stability_runs, device,
            n_epochs=n_epochs_training, bootstrap_seed=outer_bootstrap_seed, n_models=1, batch_size=64
        )
        results_current_dataset['single_nn']['stabilities'] = [s_single]
        results_current_dataset['single_nn']['accuracies'] = [a_single]
        results_current_dataset['single_nn']['stabilities_std'] = [s_single_std]
        results_current_dataset['single_nn']['accuracies_std'] = [a_single_std]
        results_current_dataset['single_nn']['disagreements_hist_data'] = d_list_single
        print(f"Single NN ({dataset_id_name}) -> Stab: {s_single:.4f}±{s_single_std:.4f}, Acc: {a_single:.3f}±{a_single_std:.3f}, Time: {time.time()-start_time:.1f}s")

        for n_k_models in n_models_ensemble_list:
            print(f"\n--- {dataset_id_name}: Evaluating with K={n_k_models} base models ---")
            methods_to_run = [
                ('standard', {}),
                ('sheaf_random_proj', {'proj_dim': input_dim_processed, 'glue_weight': common_glue_weight}),
                ('sheaf_identity_proj', {'glue_weight': common_glue_weight}),
                ('hybrid_sheaf_bagging', {'glue_weight': common_glue_weight, 'glue_subsample_ratio': hybrid_glue_subsample_ratio}),
                ('simple_ensemble_nn', {})
            ]
            for method_name, params in methods_to_run:
                print(f"Method: {method_name} for {dataset_id_name} (K={n_k_models})")
                start_time = time.time()
                s, a, d_list, s_std, a_std = calculate_stability_and_accuracy(
                    method_name, X_test, y_test, X_train, y_train,  # Test for eval, train for training
                    input_dim_processed, num_classes,
                    n_bootstrap_stability_runs, device,
                    n_models=n_k_models, n_epochs=n_epochs_training,
                    bootstrap_seed=outer_bootstrap_seed, batch_size=32, **params
                )
                results_current_dataset[method_name]['stabilities'].append(s)
                results_current_dataset[method_name]['accuracies'].append(a)
                results_current_dataset[method_name]['stabilities_std'].append(s_std)
                results_current_dataset[method_name]['accuracies_std'].append(a_std)
                
                # Store disagreements hist data for every K
                results_current_dataset[method_name]['disagreements_hist_data'] = d_list
                
                print(f"{method_name} (K={n_k_models}, {dataset_id_name}) -> Stab: {s:.4f}±{s_std:.4f}, Acc: {a:.3f}±{a_std:.3f}, Time: {time.time()-start_time:.1f}s")

            # Statistical tests for this K
            method_stab_means = {}
            for method_name, _ in methods_to_run:
                idx = n_models_ensemble_list.index(n_k_models)
                if len(results_current_dataset[method_name]['stabilities']) > idx:
                    stab_mean = results_current_dataset[method_name]['stabilities'][idx]
                    method_stab_means[method_name] = stab_mean

            method_names = list(method_stab_means.keys())
            for i in range(len(method_names)):
                for j in range(i+1, len(method_names)):
                    m1, m2 = method_names[i], method_names[j]
                    d1 = results_current_dataset[m1]['disagreements_hist_data']
                    d2 = results_current_dataset[m2]['disagreements_hist_data']
                    if d1 is not None and d2 is not None and len(d1) > 0 and len(d2) > 0:
                        stat, pval = mannwhitneyu(d1, d2, alternative='two-sided')
                        if pval < 0.05:
                            print(f"  [K={n_k_models}] Significant stability difference between {m1} and {m2}: p={pval:.4g}")

            # Clear detailed results before next K
            for method_name in results_current_dataset:
                results_current_dataset[method_name]['disagreements_hist_data'] = None

        # Collect differences for this dataset
        acc_single_nn_val = results_current_dataset['single_nn']['accuracies'][0] if results_current_dataset['single_nn']['accuracies'] else np.nan
        stab_single_nn_val = results_current_dataset['single_nn']['stabilities'][0] if results_current_dataset['single_nn']['stabilities'] else np.nan

        for method_name_iter in results_current_dataset:
            if method_name_iter == 'single_nn': continue
            for k_idx, k_val in enumerate(n_models_ensemble_list):
                if k_idx < len(results_current_dataset[method_name_iter]['accuracies']):
                    acc_method = results_current_dataset[method_name_iter]['accuracies'][k_idx]
                    stab_method = results_current_dataset[method_name_iter]['stabilities'][k_idx]
                    acc_diff = acc_method - acc_single_nn_val if not (np.isnan(acc_method) or np.isnan(acc_single_nn_val)) else np.nan
                    stab_diff = stab_method - stab_single_nn_val if not (np.isnan(stab_method) or np.isnan(stab_single_nn_val)) else np.nan
                    all_dataset_method_diff_results.append({
                        'dataset': dataset_id_name,
                        'method': method_name_iter,
                        'K': k_val,
                        'acc_diff': acc_diff,
                        'stab_diff': stab_diff,
                    })

        # Save per-method/K results for this dataset
        for method_name in results_current_dataset:
            if method_name == 'single_nn':
                accs = [(k, results_current_dataset[method_name]['accuracies'][0]) for k in n_models_ensemble_list]
                stabs = [(k, results_current_dataset[method_name]['stabilities'][0]) for k in n_models_ensemble_list]
            else:
                accs = list(zip(n_models_ensemble_list, results_current_dataset[method_name]['accuracies']))
                stabs = list(zip(n_models_ensemble_list, results_current_dataset[method_name]['stabilities']))
            accuracies_by_dataset[dataset_id_name][method_name] = accs
            stabilities_by_dataset[dataset_id_name][method_name] = stabs

    # Print overall results
    print("\nAccuracies by dataset/method/K:")
    for ds, methods in accuracies_by_dataset.items():
        print(f"Dataset: {ds}")
        for method, k_acc_list in methods.items():
            print(f"  {method}: {k_acc_list}")

    print("\nStabilities by dataset/method/K:")
    for ds, methods in stabilities_by_dataset.items():
        print(f"Dataset: {ds}")
        for method, k_stab_list in methods.items():
            print(f"  {method}: {k_stab_list}")

    # Save results as JSON
    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("accuracies_by_dataset_real.json", "w") as f:
        json.dump(accuracies_by_dataset, f, indent=2, default=convert_np)
    with open("stabilities_by_dataset_real.json", "w") as f:
        json.dump(stabilities_by_dataset, f, indent=2, default=convert_np)

if __name__ == '__main__':
    main()