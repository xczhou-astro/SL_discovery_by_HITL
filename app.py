import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
import matplotlib
from scipy.interpolate import interp1d

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--seed', type=int, default=4321)
args = parser.parse_args()
results_path = args.results_path
seed = args.seed
            
app = Flask(__name__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS (Apple Silicon) for training.')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA for training.')
else:
    device = torch.device('cpu')
    print('Using CPU for training.')
    

class SpatialGatingUnit(nn.Module):
    """
    Applies a spatial gating mechanism to an input tensor.

    This unit splits the input tensor along the channel dimension and uses one half
    to gate the other half after a linear projection. This allows for interactions

    between different 'spatial' locations (here, we treat feature dimensions as spatial).
    """
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        # Project along the sequence/feature dimension
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_ffn]
        # Split into two halves along the last dimension (d_ffn)
        u, v = x.chunk(2, dim=-1)

        # Apply gating: v is normalized and projected to learn spatial interactions
        v = self.norm(v)
        # Transpose to [batch_size, d_ffn/2, seq_len] for projection
        v = v.transpose(1, 2)
        v = self.proj(v)
        # Transpose back to [batch_size, seq_len, d_ffn/2]
        v = v.transpose(1, 2)

        # Element-wise multiplication to gate u
        return u * v

class gMLPBlock(nn.Module):
    """
    A single gMLP block which includes normalization, channel projections,
    and a spatial gating unit.
    """
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Project up to the feed-forward dimension
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        # The core spatial gating unit
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        # Project back down to the model dimension
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]
        residual = x
        x = self.norm(x)
        
        # Apply GELU activation after the first projection
        x = F.gelu(self.channel_proj1(x))
        
        # Apply the spatial gating
        x = self.sgu(x)
        
        # Apply the final projection
        x = self.channel_proj2(x)
        
        # Add the residual connection
        return x + residual

class LatentClassifier(nn.Module):
    """
    Classifier using a gMLP block for a single latent vector input.
    """
    def __init__(self, input_dim=384, d_ffn_factor=2, depth=2, bayesian=False, dropout_rate=0.2):
        super().__init__()
        # We treat the input dimension as the sequence length
        self.seq_len = input_dim
        # We treat the model dimension as 1 for simplicity, as we have only one vector
        self.d_model = 1
        d_ffn = self.d_model * d_ffn_factor
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # Stack multiple gMLP blocks if needed
        self.gmlp_layers = nn.Sequential(
            *[gMLPBlock(self.d_model, d_ffn, self.seq_len) for _ in range(depth)]
        )

        # Classification head
        self.pooler = nn.Linear(input_dim, input_dim)
        
        if self.bayesian:
            self.dropout1 = nn.Dropout(dropout_rate)  # After gMLP
            self.dropout2 = nn.Dropout(dropout_rate)  # After pooler
            
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Input x has shape [batch_size, input_dim]
        
        # 1. Reshape for gMLP: [batch_size, input_dim] -> [batch_size, input_dim, 1]
        # This treats the feature dimension (input_dim) as the "sequence".
        x = x.unsqueeze(-1)

        # 2. Pass through the gMLP block(s)
        gmlp_output = self.gmlp_layers(x)

        # 3. Reshape back to original format: [batch_size, input_dim, 1] -> [batch_size, input_dim]
        x = gmlp_output.squeeze(-1)
        
        # 4. Apply dropout after gMLP
        if self.bayesian:
            x = self.dropout1(x)

        # 5. Final classification head
        x = F.gelu(self.pooler(x))
        
        # 6. Apply dropout after pooler
        if self.bayesian:
            x = self.dropout2(x)
            
        logits = self.classifier(x)

        # 7. Apply sigmoid for binary classification and squeeze to [batch_size]
        return torch.sigmoid(logits).squeeze(-1) 
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        inputs: predicted probabilities (after sigmoid), shape [batch_size]
        targets: true labels (0 or 1), shape [batch_size]
        """
        # Clamp to prevent log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Calculate focal loss
        bce_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Modulating factor
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()
    
class SLDetector:
    
    def __init__(self, data_path='/Users/xczhou/Desktop/slwork/new_data/embeddings.npz', 
                images_path='/Users/xczhou/Desktop/slwork/new_data/JWST_images',
                dataframe_path='/Users/xczhou/Desktop/slwork/new_data/JWST_SL_discovery_catalog.csv',
                embedding_size=384):
        
        self.data_path = data_path
        self.images_path = images_path
        self.dataframe_path = dataframe_path
        self.embedding_size = embedding_size
        
        self.norm_method = 'layer' # layer or batch
        self.loss_method = 'bce' # focal or bce
        
        os.makedirs(results_path, exist_ok=True)
        
        self.random_seed = seed
        self.filter = True
        self.latents_scaled = False
        
        self.ensemble_size = 5
        self.ensembles = []
        self.supplement_ratio = 0.3
        self.num_submission_train = 30
        
        config = {
            'random_seed': self.random_seed,
            'filter': self.filter,
            'latents_scaled': self.latents_scaled,
            'ensemble_size': self.ensemble_size,
            'supplement_ratio': self.supplement_ratio,
            'num_submission_train': self.num_submission_train,
            'norm_method': self.norm_method,
            'loss_method': self.loss_method,
            'embedding_size': self.embedding_size,
        }
        
        with open(f'{results_path}/config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        if self.loss_method == 'focal':
            
            print('Use focal loss')
            
            self.criterion = FocalLoss(alpha=0.25, gamma=1.0)
        elif self.loss_method == 'bce':
            
            print('Use BCE loss')
            
            self.criterion = nn.BCELoss()
            
        self.current_round = 0
        self.selected_sl_names = []
        self.selected_non_sl_names = []
        self.total_submissions = 0  # Counter for total submissions
        self.increment = 0
        self.last_sl_count = 0
        
        self.num_selected_sl_history = []
        
        self.available_names = []
        
        self.load_data()
    
        self.model_trained = False
        self.scores = {}
        self.dividing_threshold = 0
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Initialize stats dictionary to avoid AttributeError
        self.stats = {
            'min_score': 0.0,
            'min_score_cowls': 0.0,
            'num_over_min_score': 0,
            'num_over_min_score_cowls': 0,
        }
        
        self.load_history()            
        
    def load_data(self):
        
        self.df = pd.read_csv(self.dataframe_path)
        df_names = self.df['name']
        
        if self.filter:
            # consider the max magnitude of candidates from COWLS
            self.df = self.df[self.df['ABmag_F444W'] < 24.6]
            df_names = self.df['name']
        
        print(f"Final dataset: {len(df_names)} total images")
        
        self.cowls_sl_names = self.df[self.df['COWLS'] == 1]['name'].tolist()
        self.cowls_sl_grades = self.df[self.df['COWLS'] == 1]['grade'].tolist()
        
        self.name_to_grade = {name: grade for name, grade in zip(self.cowls_sl_names, self.cowls_sl_grades)}
        
        self.selected_sl_names += self.cowls_sl_names
        self.last_sl_count = len(self.selected_sl_names)
        
        print('Number of initial SL names: ', len(self.selected_sl_names))

        data = np.load(self.data_path)
        self.latents = data['embeddings']
        self.galaxy_names = data['names']
        
        intersect_names, gal_idx, fil_idx = np.intersect1d(self.galaxy_names, df_names, return_indices=True)
        self.latents = self.latents[gal_idx].astype(np.float32)
        
        if self.latents_scaled:
            
            print('Scale latents using standard scaler')
            
            scaler = StandardScaler()
            self.latents = scaler.fit_transform(self.latents)
            self.latents = self.latents.astype(np.float32)
            
            with open(os.path.join(self.results_path, 'scaler.pkl'), 'wb') as f:
                joblib.dump(scaler, f)
        
        self.galaxy_names = self.galaxy_names[gal_idx]
        self.df = self.df.iloc[fil_idx]
        
        self.name_to_idx = {name: idx for name, idx in zip(self.galaxy_names, range(len(self.galaxy_names)))}
        
    def get_available_galaxies(self):
        
        # exclude selected SL and non-SL
        excluded_names = set(self.selected_sl_names + self.selected_non_sl_names)
        excluded_names = np.array(list(excluded_names))
        
        self.available_names = np.setdiff1d(self.galaxy_names, excluded_names)
        
        return len(self.available_names)
    
    def get_random_batch(self, size=10):
        
        self.get_available_galaxies()
        
        available_size = min(size, len(self.available_names))
        if available_size == 0:
            return [], []
        
        np.random.seed(self.random_seed)
        selected_names = np.random.choice(self.available_names, available_size, replace=False)
        
        if self.model_trained and self.scores:
            selected_scores = [self.scores.get(name, 0.5) for name in selected_names]
        else:
            selected_scores = [0.5] * available_size
            
        return selected_names.tolist(), selected_scores
    
    def add_selections(self, sl_names, non_sl_names):
        
        # Count only NEW selections (not duplicates)
        new_sl_count = len([name for name in sl_names if name not in self.selected_sl_names])
        new_non_sl_count = len([name for name in non_sl_names if name not in self.selected_non_sl_names])
        
        # Add new items while preserving order
        for name in sl_names:
            if name not in self.selected_sl_names:
                self.selected_sl_names.append(name)
        
        for name in non_sl_names:
            if name not in self.selected_non_sl_names:
                self.selected_non_sl_names.append(name)
        
        # Increment submission counter by the number of NEW selections only
        self.total_submissions += new_sl_count + new_non_sl_count
        
        print(f'Added {new_sl_count} new SL and {new_non_sl_count} new non-SL selections')
        print(f'Total submissions: {self.total_submissions}')
    
    def parallel_training(self, model, dataloader):
        
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        training_losses = []
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        epochs = 300
        
        for epoch in range(epochs):
            
            model.train()
            train_epoch_loss = 0
            
            for batch_latents, batch_labels in dataloader:
                
                batch_latents = batch_latents.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                batch_predictions = model(batch_latents).squeeze()
                batch_labels = batch_labels.squeeze()
                loss = self.criterion(batch_predictions, batch_labels)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                
            avg_loss = train_epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # best_epoch = epoch + 1
                # print(f'Best loss updated at epoch {best_epoch}: {best_loss:.4f}')
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if epoch >= 100: 
                if patience_counter >= patience:
                    # print(f"Early stopping at epoch {epoch}")
                    break
                
        model.load_state_dict(best_model_state)
        
        return model, training_losses
    
    def train_model(self, epochs=300):
        
        # Create and add new model first
        model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                 bayesian=False)
        self.ensembles.append(model)
        
        # Remove oldest if exceeded ensemble size
        if len(self.ensembles) > self.ensemble_size:
            print(f'Ensemble size exceeded {self.ensemble_size}. Removing oldest model...')
            self.ensembles.pop(0)
        
        sl_count = len(self.selected_sl_names)
        non_sl_count = len(self.selected_non_sl_names)
        total_count = sl_count + non_sl_count
        
        print(f"\n=== Training ensembles - Round {self.current_round} ===")
        print(f'Ensemble size: {len(self.ensembles)}')
        print(f"SL sources: {sl_count}")
        print(f"Non-SL sources: {non_sl_count}")
        print(f"Total training samples: {total_count}")
        print(f"SL ratio: {sl_count/total_count:.2%}")
        print("=" * 40)
        
        training_names = np.array(self.selected_sl_names + self.selected_non_sl_names)
        training_labels = np.array([1] * len(self.selected_sl_names) + [0] * len(self.selected_non_sl_names))
        
        training_indices = np.array([self.name_to_idx[name] for name in training_names])
        training_latents = self.latents[training_indices]
        
        print(f'Using {self.loss_method} loss function')
        
        if self.loss_method == 'focal':
            
            dataset = TensorDataset(
                torch.FloatTensor(training_latents),
                torch.FloatTensor(training_labels)
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
        elif self.loss_method == 'bce':
            
            class_counts = np.bincount(training_labels)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in training_labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            # Create dataset
            dataset = TensorDataset(
                torch.FloatTensor(training_latents),
                torch.FloatTensor(training_labels)
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)
        
        print('Parallel training...')
        
        outputs = joblib.Parallel(n_jobs=len(self.ensembles))(
            joblib.delayed(self.parallel_training)(model, dataloader)
            for model in self.ensembles
        )
        
        print('Parallel training - done')
        
        models, ensemble_training_losses = zip(*outputs)
        self.ensembles = list(models)
        
        self.model_trained = True
        self.round_save_path = os.path.join(f'{results_path}', 
                                            f'round_{self.current_round}')
        os.makedirs(self.round_save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        ensemble_losses = {}
        for i, loss in enumerate(ensemble_training_losses):
            ensemble_losses[f'model_{i}'] = loss
            plt.plot(loss, label=f'model_{i}', c=self.colors[i])
            min_epoch = np.argmin(loss)
            min_loss = np.min(loss)
            plt.axvline(x=min_epoch, linestyle='-.', c=self.colors[i], alpha=0.5,
                        label=f'model_{i} best epoch {min_epoch} ({min_loss:.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.round_save_path, 'ensemble_losses.png'))
        plt.close()
            
        with open(os.path.join(self.round_save_path, 'ensemble_losses.json'), 'w') as f:
            json.dump(ensemble_losses, f, indent=4)
            
        self.update_scores()
        
        self.sl_scores = [self.scores[name] for name in self.selected_sl_names]
        self.non_sl_scores = [self.scores[name] for name in self.selected_non_sl_names]
        
        self.dividing_threshold = self.calculate_dividing_threshold(self.sl_scores, self.non_sl_scores)
        self.create_visualizations()
        self.save_records()
        self.save_model()
        
        self.current_round += 1
        
        return {
            'success': True,
            'round': self.current_round,
            'epochs': epochs,
            'training_samples': len(training_names),
            'sl_count': len(self.selected_sl_names),
            'non_sl_count': len(self.selected_non_sl_names),
            'available_count': self.get_available_galaxies(),
            'model_trained': self.model_trained,
        }
        
    def parallel_scoring(self, model, dataloader):
        
        model.to(device)
        model.eval()
        
        scores = []
        
        for batch_latents in dataloader:
            batch_latents = batch_latents.to(device)
            with torch.no_grad():
                batch_scores = model(batch_latents).squeeze()
            scores.append(batch_scores.cpu().numpy().flatten())
            
        scores = np.concatenate(scores)
            
        return scores
    
    def update_scores(self):
        
        self.increment = len(self.selected_sl_names) - self.last_sl_count
        
        # increment of selected SL in previous 5 rounds
        self.num_selected_sl_history.append(self.increment)
        if len(self.num_selected_sl_history) > 5:
            self.num_selected_sl_history.pop(0)
        
        self.last_sl_count = len(self.selected_sl_names)
        
        print('Increment SL sources: ', self.increment)
        
        print('Updating scores...')
        
        if not self.model_trained:
            self.scores = {}
            return
        
        self.get_available_galaxies()
        
        all_names = self.galaxy_names
        all_indices = np.array([self.name_to_idx[name] for name in all_names])
        all_latents = self.latents[all_indices]
        
        dataloader = DataLoader(torch.FloatTensor(all_latents), 
                                batch_size=256, shuffle=False)
        
        
        print('Getting score using joblib.Parallel')
        
        ensemble_scores = joblib.Parallel(n_jobs=len(self.ensembles))(
            joblib.delayed(self.parallel_scoring)(model, dataloader)
            for model in self.ensembles
        )
        
        print('Getting score using joblib.Parallel - done')
        
        ensemble_scores = np.array(ensemble_scores)
        scores = np.mean(ensemble_scores, axis=0)
        
        self.scores = {name: score for name, score in zip(all_names, scores)}
        
        dataframe = self.df.copy()
        
        results = {}
        results['name'] = list(self.scores.keys())
        results['score'] = list(self.scores.values())
        
        # with name, score and std
        df = pd.DataFrame(results)
        selected_sl = df['name'].isin(self.selected_sl_names)
        df['selected_sl'] = selected_sl
        df['selected_sl'] = df['selected_sl'].astype(int)
        
        selected_non_sl = df['name'].isin(self.selected_non_sl_names)
        df['selected_non_sl'] = selected_non_sl
        df['selected_non_sl'] = df['selected_non_sl'].astype(int)
        
        df = pd.merge(dataframe, df, on='name', how='left')
        df.sort_values(by=['COWLS', 'score'], ascending=[False, False], inplace=True)
        df.to_csv(os.path.join(self.round_save_path, 'scores.csv'), index=False)
        
        print(f'Updated scores for {len(self.scores)} galaxies')
        
    def save_model(self):
        print('Saving model...')
        
        ensemble_model = {}
        for i in range(len(self.ensembles)):
            ensemble_model[f'model_{i}'] = self.ensembles[i].state_dict()
        
        torch.save(ensemble_model, os.path.join(self.round_save_path, 'model.pth'))

    def custom_serializer(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Type {type(obj)} not serializable')
    
    def save_records(self):
        print('Saving records...')
        records = {
            'round': self.current_round,
            'sl_count': len(self.selected_sl_names),
            'non_sl_count': len(self.selected_non_sl_names),
            'total_submissions': self.total_submissions,
            'increment_sl_count': self.increment,
            'mean_num_selected_sl_history': np.mean(self.num_selected_sl_history).item(),
            'model_trained': self.model_trained,
            'num_ensembles': len(self.ensembles),
            'dividing_threshold': self.dividing_threshold,
            'min_score': self.stats['min_score'],
            'num_over_min_score': self.stats['num_over_min_score'],
            'min_score_cowls': self.stats['min_score_cowls'],
            'num_over_min_score_cowls': self.stats['num_over_min_score_cowls'],
            'num_lower_max_non_sl_score': self.stats['num_lower_max_non_sl_score'],
            'purity': len(self.selected_sl_names) / self.stats['num_over_min_score_cowls'],
            'sl_names': self.selected_sl_names,
            'non_sl_names': self.selected_non_sl_names,
        }
        
        with open(os.path.join(self.round_save_path, 'records.json'), 'w') as f:
            json.dump(records, f, indent=4, default=self.custom_serializer)

    
    def calculate_dividing_threshold(self, sl_scores, non_sl_scores):
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        scores = np.array(sl_scores + non_sl_scores).reshape(-1, 1)
        gmm.fit(scores)
        
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_
        
        threshold = np.mean(means)
        
        print(f"Dividing threshold: {threshold}")
        
        return threshold
    
    def create_visualizations(self):
        
        print('Creating visualizations...')
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        all_scores = list(self.scores.values())
        all_scores = np.array(all_scores)
        max_score = np.max(all_scores)
        
        thresholds = np.linspace(0, 1, 100)
        counts = np.array([np.sum(all_scores >= t) for t in thresholds])
        
        interp = interp1d(thresholds, counts, kind='cubic')
        
        # COWLS
        scores_for_cowls_sl = [self.scores[name] for name in self.cowls_sl_names]
        interp_counts_cowls_sl = [interp(score) for score in scores_for_cowls_sl]
        
        sl_names_exclude_cowls = [name for name in self.selected_sl_names if name not in self.cowls_sl_names]
        scores_for_sl_exclude_cowls = [self.scores[name] for name in sl_names_exclude_cowls]
        interp_counts_sl_exclude_cowls = [interp(score) for score in scores_for_sl_exclude_cowls]
        
        scores_for_non_sl = [self.scores[name] for name in self.selected_non_sl_names]
        max_score_non_sl = np.max(scores_for_non_sl)
        
        # min score - handle empty lists
        min_scores = []
        if len(scores_for_cowls_sl) > 0:
            min_scores.append(np.min(scores_for_cowls_sl))
        if len(scores_for_sl_exclude_cowls) > 0:
            min_scores.append(np.min(scores_for_sl_exclude_cowls))
            
        min_score = np.min(min_scores) if len(min_scores) > 0 else 0.0
        min_score_cowls = np.min(scores_for_cowls_sl) if len(scores_for_cowls_sl) > 0 else 0.0
        
        num_over_threshold = np.sum(all_scores > self.dividing_threshold)
        num_over_min_score = np.sum(all_scores > min_score)
        num_over_min_score_cowls = np.sum(all_scores > min_score_cowls)

        num_lower_max_non_sl_score = np.sum(all_scores < max_score_non_sl)
        
        self.stats = {
            'min_score': min_score,
            'min_score_cowls': min_score_cowls,
            'num_over_min_score': num_over_min_score,
            'num_over_min_score_cowls': num_over_min_score_cowls,
            'num_lower_max_non_sl_score': num_lower_max_non_sl_score,
        }
        
        print('stats: ', self.stats)
        
        bins = np.arange(0, 1.02, 0.02)
        # hist, bin_edges = np.histogram(all_scores, bins=bins)
        # percentages = (hist / len(all_scores))
        # x_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        hist_sl, bin_edges_sl = np.histogram(self.sl_scores, bins=bins)
        percentages_sl = (hist_sl / len(self.sl_scores))
        x_bins_sl = (bin_edges_sl[:-1] + bin_edges_sl[1:]) / 2
        
        hist_non_sl, bin_edges_non_sl = np.histogram(self.non_sl_scores, bins=bins)
        percentages_non_sl = (hist_non_sl / len(self.non_sl_scores))
        x_bins_non_sl = (bin_edges_non_sl[:-1] + bin_edges_non_sl[1:]) / 2
        
        axes[0].bar(x_bins_sl, percentages_sl, width=0.02, alpha=0.7, color='skyblue', edgecolor='black',
                    label='SL')
        axes[0].bar(x_bins_non_sl, percentages_non_sl, width=0.02, alpha=0.7, color='orange', edgecolor='black',
                    label='Non-SL')
        axes[0].axvline(max_score, color='k', linestyle=':', linewidth=2,
                        label=f'Max Score: {max_score:.3f}')
        axes[0].axvline(self.dividing_threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Dividing Threshold: {self.dividing_threshold:.3f}')
        axes[0].axvline(min_score, color='purple', linestyle='-.', linewidth=2,
                        label=f'Min Score: {min_score:.3f}')
        axes[0].axvline(min_score_cowls, color='orange', linestyle=':', linewidth=2,
                        label=f'Min score (COWLS): {min_score_cowls:.3f}')
        axes[0].set_xlabel('Scores')
        axes[0].set_ylabel('Percentage')
        axes[0].set_title(f'Score Distribution - Round {self.current_round}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(counts, thresholds, 'b-', linewidth=2)
        
        axes[1].axhline(self.dividing_threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Dividng threshold: {self.dividing_threshold:.3f} ({num_over_threshold})')
        axes[1].axhline(min_score, color='purple', linestyle='-.', linewidth=2,
                         label=f'Min score (SL): {min_score:.3f} ({num_over_min_score})')
        axes[1].axhline(min_score_cowls, color='orange', linestyle=':', linewidth=2, 
                        label=f'Min score (COWLS): {min_score_cowls:.3f} ({num_over_min_score_cowls})')
        axes[1].axhline(max_score_non_sl, color='green', linestyle=':', linewidth=2,
                        label=f'Max score (non-SL): {max_score_non_sl:.3f} ({num_lower_max_non_sl_score})')
        
        axes[1].scatter(interp_counts_cowls_sl, scores_for_cowls_sl, color='red', marker='o', s=30,
                        label=f'COWLS candidates ({len(self.cowls_sl_names)})')
        axes[1].scatter(interp_counts_sl_exclude_cowls, scores_for_sl_exclude_cowls, color='purple', marker='*', s=30,
                        label=f'Selected SL candidates ({len(sl_names_exclude_cowls)})')

        axes[1].set_xlabel('Number of samples')
        axes[1].set_ylabel('Scores')
        axes[1].set_title(f'Scores vs Number of Samples - Round {self.current_round}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.round_save_path, 'visualizations.png'))
        plt.close()
        
        print('Visualizations created.')
        
    def get_images(self):
        
        self.get_available_galaxies()
        if len(self.available_names) == 0:
            print('No available images')
            return [], []
        
        available_names = np.array(self.available_names)
        available_scores = np.array([self.scores[name] for name in available_names])
        
        idx = np.argsort(available_scores)[::-1] # from high to low
        sorted_names = available_names[idx]
        
        num_images = 10
        
        if self.supplement_ratio > 0:
            print('Get some supplement images by ratio: ', self.supplement_ratio)
            
            num_supplement = int(num_images * self.supplement_ratio)
            num_high_score = num_images - num_supplement
            
            print('Number high score: ', num_high_score)
            print('Number supplement: ', num_supplement)
            
            high_names = sorted_names[:num_high_score]
            high_scores = np.array([self.scores[name] for name in high_names])
        
            print('High score names: ', high_names)
            print('High scores: ', high_scores)
            
            remaining_names = sorted_names[num_high_score:]
            remaining_scores = np.array([self.scores[name] for name in remaining_names])
            
            idx = np.argsort(np.abs(remaining_scores - self.dividing_threshold))
            remaining_names = remaining_names[idx]
            remaining_scores = remaining_scores[idx]
            
            supplement_names = remaining_names[:num_supplement]
            supplement_scores = remaining_scores[:num_supplement]
            
            print('Supplement names: ', supplement_names)
            print('Supplement scores: ', supplement_scores)
            
            selected_names = np.concatenate([high_names, supplement_names])
            selected_scores = np.concatenate([high_scores, supplement_scores])

        else:
            selected_names = sorted_names[:num_images]
            selected_scores = np.array([self.scores[name] for name in selected_names])
        
        return selected_names.tolist(), selected_scores.tolist()
    
    def load_history(self):
        round_dirs = os.listdir(f'{results_path}')
        round_dirs = [name for name in round_dirs if 'round' in name]
        files = ['ensemble_losses.json', 'ensemble_losses.png', 
                 'model.pth', 'records.json', 'scores.csv', 'visualizations.png']
        
        if len(round_dirs) == 0:
            print('No history found. Start from fresh.')
            
        else:
            round_count = [int(name.split('_')[1]) for name in round_dirs]
            max_round = max(round_count)
            
            round_path = os.path.join(f'{results_path}', f'round_{max_round}')
            
            for file in files:
                file_path = os.path.join(round_path, file)
                if not os.path.exists(file_path):
                    raise Exception(f'File {file} not found in last round {max_round}.')
            
            
            with open(os.path.join(round_path, 'records.json'), 'r') as f:
                records = json.load(f)
            
            self.current_round = records['round']
            self.selected_sl_names = records['sl_names']
            self.selected_non_sl_names = records['non_sl_names']
            self.total_submissions = records.get('total_submissions')
            self.model_trained = records['model_trained']
            self.dividing_threshold = records['dividing_threshold']
            self.increment = records['increment_sl_count']
            self.round_save_path = round_path
            num_ensembles = records['num_ensembles']
            self.num_selected_sl_history = records['mean_num_selected_sl_history']
            
            self.stats['min_score'] = records['min_score']
            self.stats['min_score_cowls'] = records['min_score_cowls']
            self.stats['num_over_min_score'] = records['num_over_min_score']
            self.stats['num_over_min_score_cowls'] = records['num_over_min_score_cowls']
                
            weights_path = os.path.join(round_path, 'model.pth')
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            
            # Load ensemble models
            self.ensembles = []
            
            for i in range(num_ensembles):
                
                model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2)
                
                model_state = checkpoint[f'model_{i}']
                model.load_state_dict(model_state)
            
                self.ensembles.append(model)
                
            print(f'Successfully loaded model weights of {num_ensembles} ensembles.')
            
            self.get_available_galaxies()
            
            df_scores = pd.read_csv(os.path.join(round_path, 'scores.csv'))
            self.scores = {name: float(score) for name, score in zip(df_scores['name'], df_scores['score'])}
            
            print(f'Load history from {round_path}.')
            print('Current round: ', self.current_round)

            self.current_round += 1

sl_detector = SLDetector()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_random_batch')
def api_get_random_batch():
    names, scores = sl_detector.get_random_batch(10)
    
    try:
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores, 
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error geting random batch.')
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_images')
def api_get_images():
    try:
        print(f'\n=== /api/get_images called ===')
        print(f'Model trained: {sl_detector.model_trained}')
        
        if sl_detector.model_trained:
            print('Using get_images() (smart selection)')
            names, scores = sl_detector.get_images()
        else:
            print('Using get_random_batch() (random selection)')
            names, scores = sl_detector.get_random_batch(10)
        
        print(f'Returning {len(names)} images')
        
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error getting images:', str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/app/submit_selections', methods=['POST'])
def submit_selections():
    
    print('\n=== /app/submit_selections called ===')
    
    try:
        data = request.json
        sl_names = data.get('sl_names', [])
        non_sl_names = data.get('non_sl_names', [])
        mode = data.get('mode', 'random')
        
        print(f'Received SL names: {sl_names}')
        print(f'Received Non-SL names: {non_sl_names}')
        print(f'Mode: {mode}')
        
        sl_detector.add_selections(sl_names, non_sl_names)
        
        print(f'After add_selections - SL count: {len(sl_detector.selected_sl_names)}, Non-SL count: {len(sl_detector.selected_non_sl_names)}')
        num_submission_train = sl_detector.num_submission_train
        
        response_data = {
            'success': True,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
        }
        
        # Check if auto-training should be triggered (every 100 submissions)
        if sl_detector.total_submissions % num_submission_train == 0 and sl_detector.total_submissions > 0:
            response_data['should_train'] = True
        else:
            response_data['should_train'] = False
            # Load next batch of images
            if sl_detector.model_trained:
                names, scores = sl_detector.get_images()
            else:
                names, scores = sl_detector.get_random_batch(10)
            response_data['galaxy_names'] = names
            response_data['scores'] = scores
            response_data['model_trained'] = sl_detector.model_trained
        
        return jsonify(response_data)
    except Exception as e:
        print('Error submitting selections.')
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_training', methods=['POST'])
def run_training():
    
    epochs = request.json.get('epochs', 300)
    
    try:
        result = sl_detector.train_model(epochs)
        
        return jsonify(result)
    except Exception as e:
        print('Error running training.')
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get_status')
def get_status():
    return jsonify({
        'success': True,
        'round': sl_detector.current_round,
        'sl_count': len(sl_detector.selected_sl_names),
        'non_sl_count': len(sl_detector.selected_non_sl_names),
        'total_submissions': sl_detector.total_submissions,
        'available_count': sl_detector.get_available_galaxies(),
        'model_trained': sl_detector.model_trained,
    })

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/api/get_gallery_data')
def get_gallery_data():
    try:
        print('\n=== /api/get_gallery_data called ===')
        
        # Separate confirmed and user-selected
        confirmed_sl_names = set(sl_detector.cowls_sl_names)
        
        # Preserve selection order for user-selected items
        user_selected_sl_names = [name for name in sl_detector.selected_sl_names if name not in confirmed_sl_names]
        user_selected_non_sl_names = sl_detector.selected_non_sl_names
        
        print(f'Confirmed SL: {len(confirmed_sl_names)}')
        print(f'User-selected SL: {len(user_selected_sl_names)}')
        print(f'User-selected non-SL: {len(user_selected_non_sl_names)}')
        if len(user_selected_non_sl_names) > 0:
            print(f'First 3 non-SL names: {user_selected_non_sl_names[:3]}')
        
        # Define grade order for sorting confirmed items (high to low)
        high_grades = ['M25'] + [f'S{i:02d}' for i in range(10, 0, -1)]
        grade_order = {grade: i for i, grade in enumerate(high_grades)}
        
        # Create confirmed SL items sorted by grade
        confirmed_items = []
        for name in sl_detector.cowls_sl_names:
            confirmed_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': True,
                'grade': sl_detector.name_to_grade.get(name, 'N/A')
            })
        
        # Sort confirmed items by grade (high to low)
        confirmed_items.sort(key=lambda x: (grade_order.get(x['grade'], 999), x['name']))
        
        # Create user-selected SL items in selection order
        user_selected_sl_items = []
        for name in user_selected_sl_names:
            user_selected_sl_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': False,
                'grade': sl_detector.name_to_grade.get(name, 'N/A')
            })
        
        # Create user-selected non-SL items in selection order
        user_selected_non_sl_items = []
        for name in user_selected_non_sl_names:
            user_selected_non_sl_items.append({
                'name': name,
                'type': 'non_sl',
                'is_confirmed': False,
                'grade': 'N/A'
            })
        
        # Combine: confirmed SL first, then user-selected SL, then user-selected non-SL
        gallery_items = confirmed_items + user_selected_sl_items + user_selected_non_sl_items
        
        print(f'Total gallery items: {len(gallery_items)}')
        print(f'  - Confirmed SL items: {len(confirmed_items)}')
        print(f'  - User-selected SL items: {len(user_selected_sl_items)}')
        print(f'  - User-selected non-SL items: {len(user_selected_non_sl_items)}')
        
        return jsonify({
            'success': True,
            'items': gallery_items.tolist() if isinstance(gallery_items, np.ndarray) else gallery_items,
            'total_count': len(gallery_items),
            'confirmed_count': len(confirmed_sl_names),
            'selected_sl_count': len(user_selected_sl_names),
            'selected_non_sl_count': len(user_selected_non_sl_names)
        })
    except Exception as e:
        print('Error getting gallery data:', str(e))
        return jsonify({'error': str(e)}), 500
    
@app.route('/images/<filename>')
def serve_image(filename):
    # Add .jpg extension if not present
    if not filename.endswith('.jpg'):
        filename = f"{filename}.jpg"
    return send_from_directory(sl_detector.images_path, filename)

@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    try:
        if hasattr(sl_detector, 'round_save_path') and sl_detector.round_save_path:
            return send_from_directory(sl_detector.round_save_path, filename)
        else:
            # Try to find the latest results directory
            results_dirs = [d for d in os.listdir(f'{results_path}') if d.startswith('round_')]
            if results_dirs:
                latest_dir = max(results_dirs, key=lambda x: int(x.split('_')[1]))
                return send_from_directory(os.path.join(f'{results_path}', latest_dir), filename)
            else:
                return "No visualizations available", 404
    except Exception as e:
        return f"Error serving visualization: {str(e)}", 500

if __name__ == '__main__':
    
    app.run(debug=False, host='0.0.0.0', port=6543)