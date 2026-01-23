import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.func import stack_module_state, functional_call, vmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import copy
from tqdm import tqdm
import time

from models import LatentClassifier, FocalLoss, CudaDataLoader
from configurations import config

class SLDetector:
    
    def __init__(self):
        
        # TF32 for faster training (enables TensorFloat-32 on Ampere+ GPUs)
        torch.set_float32_matmul_precision('high')
        
        self.data_path = config.data_path
        self.images_path = config.images_path
        self.dataframe_path = config.dataframe_path
        self.embedding_size = config.embedding_size
        self.results_path = config.results_path
        
        os.makedirs(self.results_path, exist_ok=True)
        
        self.norm_method = config.norm_method # layer or batch
        self.random_seed = config.random_seed
        self.filter = config.filter
        self.mag_limit = config.mag_limit
        
        self.latents_scaled = config.latents_scaled
        
        self.maximum_ensemble_size = config.maximum_ensemble_size
        
        self.supplement_ratio = config.supplement_ratio
        self.supplement_method = config.supplement_method # threshold or uncertainty
        self.num_submission_train = config.num_submission_train
        self.patience = config.patience
        self.epochs = config.epochs
        self.warmup_epochs = config.warmup_epochs
        self.batch_size = config.batch_size
        
        self.fix_ensembles = config.fix_ensembles
        
        self.criterion = nn.BCEWithLogitsLoss()
            
        self.ensembles = []
        self.current_round = 0
        self.selected_sl_names = []
        self.selected_non_sl_names = []
        self.total_submissions = 0  # Counter for total submissions
        
        self.num_increment_history = []
        self.mean_increment = 0
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.available_names = []
        
        self.model_trained = False
        self.scores = {}
        self.dividing_threshold = 0
        
        # Initialize stats dictionary to avoid AttributeError
        self.stats = {
            'min_score': 0.0,
            'min_score_cowls': 0.0,
            'num_over_min_score': 0,
            'num_over_min_score_cowls': 0,
            'num_lower_max_non_sl_score': 0,
        }
        
        if self.fix_ensembles:
            self.initialize_ensembles()
        
        self.load_data()
        self.load_history()
        
    def load_data(self):
        
        self.df = pd.read_csv(self.dataframe_path, low_memory=False)
        df_names = self.df['name']
        
        if self.filter:
            # consider the max magnitude of candidates from COWLS
            self.df = self.df[self.df['ABmag_F444W'] < self.mag_limit]
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
        
        # move all data to cuda device (non_blocking=True for async transfer)
        self.latents = torch.from_numpy(self.latents).to(config.device, non_blocking=True)
        
        print('Latent shape: ', self.latents.shape)
        
        if self.latents_scaled:
            
            print('Scale latents using standard scaler')
            
            # Convert to numpy for StandardScaler, then back to torch tensor
            latents_np = self.latents.cpu().numpy()
            scaler = StandardScaler()
            latents_np = scaler.fit_transform(latents_np)
            self.latents = torch.from_numpy(latents_np.astype(np.float32)).to(config.device, non_blocking=True)
            
            with open(os.path.join(self.results_path, 'scaler.pkl'), 'wb') as f:
                joblib.dump(scaler, f)
        
        self.galaxy_names = self.galaxy_names[gal_idx]
        self.df = self.df.iloc[fil_idx]
        
        self.name_to_idx = {name: idx for name, idx in zip(self.galaxy_names, range(len(self.galaxy_names)))}
        
    def initialize_ensembles(self):
        
        self.ensembles = []
        for i in range(self.maximum_ensemble_size):
            model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                 bayesian=False).to(config.device)
            self.ensembles.append(model)
        
        params, buffers = stack_module_state(self.ensembles)
        self.optimizer = optim.Adam(params.values(), lr=1e-3, weight_decay=1e-5, 
                                    fused=True)
        self.scaler = torch.amp.GradScaler(device=config.device)
        
        def fmodel(params, buffers, x):
            return functional_call(self.ensembles[0], (params, buffers), x)
        
        self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
        self.params = params
        self.buffers = buffers
        
        self.ensemble_size = len(self.ensembles)
        
        print(f'Initialized {self.ensemble_size} ensembles')
        
    def initialize_increasing_ensembles(self):
        
        model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                    bayesian=False).to(config.device)
        self.ensembles.append(model)
        
        if len(self.ensembles) > self.maximum_ensemble_size:
            print(f'Ensemble size exceeded {self.maximum_ensemble_size}. Removing oldest model...')
            self.ensembles.pop(0)
            
        self.ensemble_size = len(self.ensembles)
        
        print('Current ensemble size: ', self.ensemble_size)
            
        params, buffers = stack_module_state(self.ensembles)
        self.optimizer = optim.Adam(params.values(), lr=1e-3, weight_decay=1e-5, 
                                    fused=True)
        self.scaler = torch.amp.GradScaler(device=config.device)
        
        def fmodel(params, buffers, x):
            return functional_call(self.ensembles[0], (params, buffers), x)
        
        self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
        self.params = params
        self.buffers = buffers
    
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
    
    def train_model(self, epochs):
        
        increment = len(self.selected_sl_names) - self.last_sl_count
        self.num_increment_history.append(increment)
        self.last_sl_count = len(self.selected_sl_names)
        
        print('Increment SL sources: ', increment)
        
        if not self.fix_ensembles:
            self.initialize_increasing_ensembles()
        
        sl_count = len(self.selected_sl_names)
        non_sl_count = len(self.selected_non_sl_names)
        total_count = sl_count + non_sl_count
        
        print(f"\n=== Training ensembles - Round {self.current_round} ===")
        print(f'Ensemble size: {self.ensemble_size}')
        print(f"SL sources: {sl_count}")
        print(f"Non-SL sources: {non_sl_count}")
        print(f"Total training samples: {total_count}")
        print(f"SL ratio: {sl_count/total_count:.2%}")
        print("=" * 40)
        
        training_names = np.array(self.selected_sl_names + self.selected_non_sl_names)
        training_labels = np.array([1] * len(self.selected_sl_names) + [0] * len(self.selected_non_sl_names))
        training_labels = torch.from_numpy(training_labels).to(config.device, non_blocking=True)
        
        training_indices = np.array([self.name_to_idx[name] for name in training_names])
        training_indices = torch.from_numpy(training_indices).to(config.device, non_blocking=True)
        
        training_latents = self.latents[training_indices]
        
        labels, counts = torch.unique(training_labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        sample_weights = class_weights[training_labels]
        
        training_labels = training_labels.float()
        
        dataloader = CudaDataLoader(
            training_latents,
            training_labels,
            self.batch_size,
            sample_weights,
        )
        
        print('Training ensembles...')
        
        start_time = time.time()
        
        training_losses = []
        best_loss = float('inf')
        best_params = copy.deepcopy(self.params)
        best_epoch = 0
        
        for epoch in range(self.epochs):
            
            train_epoch_loss = 0
            
            for batch_latents, batch_labels in dataloader:
                
                self.optimizer.zero_grad()
                
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                
                    preds = self.predict_ensemble(self.params, self.buffers, batch_latents)
                    preds_flat = preds.reshape(-1, 1)
                    
                    labels_expanded = batch_labels.expand(len(self.ensembles), -1).reshape(-1, 1)
                    labels_expanded = labels_expanded.float()
                    
                    loss = self.criterion(preds_flat, labels_expanded)
                    
                    train_epoch_loss += loss.item()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # loss.backward()
                # self.optimizer.step()
                
            avg_loss = train_epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if epoch >= self.warmup_epochs:
                # early stopping after warmup epochs
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_params = copy.deepcopy(self.params)
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Training time: {training_time:.2f} seconds')
        
        self.best_params = best_params
        
        self.model_trained = True
        self.round_save_path = os.path.join(f'{self.results_path}', 
                                            f'round_{self.current_round}')
        os.makedirs(self.round_save_path, exist_ok=True)
        
        print(f'Saving best model...')
        checkpoint = {
            'ensemble_params': copy.deepcopy(self.best_params),
            'buffers': copy.deepcopy(self.buffers),
            'config': {
                'input_dim': self.embedding_size,
                'output_dim': 1,
                'num_ensembles': self.ensemble_size,
            }
        }
        
        torch.save(checkpoint, os.path.join(self.round_save_path, 'model.pth'))
        
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label=f'Total Losses (best epoch {best_epoch})')
        plt.axvline(x=best_epoch, linestyle='-.', color='red', alpha=0.5,
                    label=f'Best epoch {best_epoch} ({best_loss:.4f})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.round_save_path, 'training_losses.png'))
        plt.close()
        
        with open(os.path.join(self.round_save_path, 'training_losses.json'), 'w') as f:
            json.dump(training_losses, f, indent=4)
            
        self.update_scores()
        
        self.sl_scores = [self.scores[name] for name in self.selected_sl_names]
        self.non_sl_scores = [self.scores[name] for name in self.selected_non_sl_names]
        
        self.dividing_threshold = self.calculate_dividing_threshold(self.sl_scores, self.non_sl_scores)
        self.create_visualizations()
        self.save_records()
        
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
    
    def update_scores(self):
        
        print('Updating scores...')
        
        start_time = time.time()
        
        if not self.model_trained:
            self.scores = {}
            return
        
        self.get_available_galaxies()
        
        all_names = self.galaxy_names
        all_indices = np.array([self.name_to_idx[name] for name in all_names])
        all_indices = torch.from_numpy(all_indices).to(config.device, non_blocking=True)
        
        all_latents = self.latents[all_indices]
        
        dataloader = CudaDataLoader(
            all_latents,
            y_tensor=None,  # Prediction mode - only x_tensor needed
            batch_size=self.batch_size,
        )
        
        scores = []
        for batch_latents in dataloader:
            
            with torch.no_grad():
                
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    preds_logits = self.predict_ensemble(self.best_params, self.buffers, batch_latents)
                    # Convert logits to probabilities for scoring
                    preds = torch.sigmoid(preds_logits)
                    
            scores.append(preds.cpu().numpy()) # (num_ensembles, batch_size)
            
        end_time = time.time()
        updating_time = end_time - start_time
        print(f'Updating scores time: {updating_time:.2f} seconds')
                
        scores = np.concatenate(scores, axis=1) # (num_ensembles, num_samples)
        mean_scores = np.mean(scores, axis=0)
        uncertainties = np.std(scores, axis=0)
        
        self.scores = {name: score for name, score in zip(all_names, mean_scores)}
        self.uncertainties = {name: uncertainty for name, uncertainty in zip(all_names, uncertainties)}
        
        dataframe = self.df.copy()
        
        results = {}
        results['name'] = list(self.scores.keys())
        results['score'] = list(self.scores.values())
        results['uncertainty'] = list(self.uncertainties.values())
        
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
            'num_increment_history': self.num_increment_history,
            'mean_increment_previous_5_rounds': np.mean(self.num_increment_history[-5:]).item(),
            'model_trained': self.model_trained,
            'num_ensembles': self.ensemble_size,
            'dividing_threshold': self.dividing_threshold,
            'min_score': self.stats['min_score'],
            'num_over_min_score': self.stats['num_over_min_score'],
            'min_score_cowls': self.stats['min_score_cowls'],
            'num_over_min_score_cowls': self.stats['num_over_min_score_cowls'],
            'num_lower_max_non_sl_score': self.stats.get('num_lower_max_non_sl_score', 0),
            'purity': len(self.selected_sl_names) / self.stats['num_over_min_score_cowls'] if self.stats['num_over_min_score_cowls'] > 0 else 0.0,
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
        
        print()
        print('stats: ')
        for key, value in self.stats.items():
            print(f'{key}: {value}')
        print()
        
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
        available_uncertainties = np.array([self.uncertainties[name] for name in available_names])
        
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
            remaining_uncertainties = np.array([self.uncertainties[name] for name in remaining_names])
            
            if self.supplement_method == 'threshold':
            
                print('Supplement by threshold')
                idx = np.argsort(np.abs(remaining_scores - self.dividing_threshold))
                remaining_names = remaining_names[idx]
                remaining_scores = remaining_scores[idx]
                
                supplement_names = remaining_names[:num_supplement]
                supplement_scores = remaining_scores[:num_supplement]
                
                print('Supplement names: ', supplement_names)
                print('Supplement scores: ', supplement_scores)
            
            elif self.supplement_method == 'uncertainty':
                
                print('Supplement by uncertainty')
                idx = np.argsort(remaining_uncertainties)[::-1] # from high to low
                
                supplement_names = remaining_names[idx][:num_supplement]
                supplement_scores = remaining_scores[idx][:num_supplement]
                supplement_uncertainties = remaining_uncertainties[idx][:num_supplement]
                
                print('Supplement names: ', supplement_names)
                print('Supplement scores: ', supplement_scores)
                print('Supplement uncertainties: ', supplement_uncertainties)
            
            selected_names = np.concatenate([high_names, supplement_names])
            selected_scores = np.concatenate([high_scores, supplement_scores])

        else:
            selected_names = sorted_names[:num_images]
            selected_scores = np.array([self.scores[name] for name in selected_names])
        
        return selected_names.tolist(), selected_scores.tolist()
    
    def load_history(self):
        
        with open(os.path.join(self.results_path, 'config.json'), 'r') as f:
            saved_config = json.load(f)
        
        files = ['training_losses.json', 'training_losses.png', 
                 'model.pth', 'records.json', 'scores.csv', 'visualizations.png']
        
        if config.checkpoint_round is None:
            
            print('No checkpoint round specified. Start from fresh.')       
             
        else:
            round_path = os.path.join(f'{self.results_path}', f'round_{config.checkpoint_round}')
            if not os.path.exists(round_path):
                raise Exception(f'Round {config.checkpoint_round} not found.')
                
            print(f'Loading checkpoint at {round_path}.')
            
            for file in files:
                file_path = os.path.join(round_path, file)
                if not os.path.exists(file_path):
                    round_num = config.checkpoint_round
                    raise Exception(f'File {file} not found in round {round_num}.')
            
            with open(os.path.join(round_path, 'records.json'), 'r') as f:
                records = json.load(f)
            
            self.current_round = records['round']
            self.selected_sl_names = records['sl_names']
            self.selected_non_sl_names = records['non_sl_names']
            self.total_submissions = records.get('total_submissions', 0) or 0
            self.model_trained = records['model_trained']
            self.dividing_threshold = records['dividing_threshold']
            self.round_save_path = round_path
            
            self.num_increment_history = records['num_increment_history']
            self.mean_increment = records['mean_increment_previous_5_rounds']
            
            self.last_sl_count = len(self.selected_sl_names)
            
            self.stats['min_score'] = records['min_score']
            self.stats['min_score_cowls'] = records['min_score_cowls']
            self.stats['num_over_min_score'] = records['num_over_min_score']
            self.stats['num_over_min_score_cowls'] = records['num_over_min_score_cowls']
            self.stats['num_lower_max_non_sl_score'] = records.get('num_lower_max_non_sl_score', 0)
            
            weights_path = os.path.join(round_path, 'model.pth')
            checkpoint = torch.load(weights_path, map_location=config.device)
            
            params = checkpoint['ensemble_params']
            buffers = checkpoint['buffers']
            model_config = checkpoint['config']
            
            self.params = params
            self.buffers = buffers
            self.ensemble_size = model_config['num_ensembles']
            
            base_model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2).to(config.device)
            
            def fmodel(params, buffers, x):
                return functional_call(base_model, (params, buffers), x)
            
            self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
            
            # Use fused optimizer for better performance
            self.optimizer = optim.Adam(self.params.values(), lr=1e-3, weight_decay=1e-5, fused=True)
            
            # Reinitialize GradScaler for mixed precision training
            self.scaler = torch.amp.GradScaler(device=config.device)
                
            print(f'Successfully loaded model weights of {self.ensemble_size} ensembles.')
            
            self.get_available_galaxies()
            
            df_scores = pd.read_csv(os.path.join(round_path, 'scores.csv'), low_memory=False)
            self.scores = {name: float(score) for name, score in zip(df_scores['name'], df_scores['score'])}
            self.uncertainties = {name: float(uncertainty) for name, uncertainty in zip(df_scores['name'], df_scores['uncertainty'])}
            
            print(f'Loaded history from {round_path}.')
            print('Current round: ', self.current_round)

            self.current_round += 1

