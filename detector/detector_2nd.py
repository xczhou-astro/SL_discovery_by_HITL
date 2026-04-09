import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import stack_module_state, functional_call, vmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import copy
import time
import shutil
import h5py

from models import LatentClassifier, CudaDataLoader, BinaryFocalLoss
from configurations import config

class SLDetector2nd:
    
    def __init__(self):
        
        torch.set_float32_matmul_precision('high')
        
        print('Using Device: ', config.device)
        
        self.data_path = config.data_path
        self.score_file = config.score_file
        self.embedding_size = config.embedding_size
        self.results_path = config.results_path
        self.images_path = config.images_path
        
        os.makedirs(self.results_path, exist_ok=True)
        
        self.norm_method = config.norm_method
        self.random_seed = config.random_seed
        self.score_limit = config.score_limit
        self.record_file = config.record_file
        
        self.latents_scaled = config.latents_scaled
        self.maximum_ensemble_size = config.maximum_ensemble_size
        
        self.supplement_ratio = config.supplement_ratio
        self.supplement_method = config.supplement_method # threshold or uncertainty
        self.num_submission_train = config.num_submission_train
        self.patience = config.patience
        self.warmup_epochs = config.warmup_epochs
        self.batch_size = config.batch_size
        
        self.fix_ensembles = config.fix_ensembles
        self.use_focal_loss = config.use_focal_loss
        self.cold_start = config.cold_start
        
        if self.use_focal_loss:
            print('Use focal loss')
            alpha = None
            self.criterion = BinaryFocalLoss(alpha=alpha, gamma=1.0)
        else:
            print('Use BCE loss')
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.ensembles = []
        self.current_round = 0
        self.selected_sl_names = []
        self.selected_non_sl_names = []
        self.total_submissions = 0  # Counter for total submissions
        self.sl_name_incremented = []
        
        self.sample_round_added = {}  # {name: round_number}
        
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
        
        self.initialize_ensembles()
        
        self.load_data()
        self.train_model()
        # self.load_history()
        
    def load_data(self):
        
        self.df = pd.read_csv(self.score_file, low_memory=False)
        
        # filter COWLS sources
        score_order = ['M25']
        score_order += [f'S{i:02d}' for i in range(0, 13)[::-1]]
        
        if self.score_limit not in score_order:
            raise ValueError(f'Invalid score limit: {self.score_limit}')

        selected_scores = score_order[:score_order.index(self.score_limit) + 1]
        print('Consider COWLS scores: ', selected_scores)
        
        idx_cowls = (self.df['COWLS'] == 1) & self.df['grade'].isin(selected_scores)
        idx_cowls_excluded = (self.df['COWLS'] == 1) & (~self.df['grade'].isin(selected_scores))
        print('COWLS sources considered: ', idx_cowls.sum())
        print('COWLS sources excluded: ', idx_cowls_excluded.sum())
        
        self.cowls_sl_names = self.df[idx_cowls]['name'].tolist()
        self.cowls_sl_grades = self.df[idx_cowls]['grade'].tolist()
        
        self.name_to_grade = {name: grade for name, grade in 
                              zip(self.cowls_sl_names, self.cowls_sl_grades)}
        
        # dataframe excluding COWLS sources that are not in the selected scores
        self.df = self.df[~idx_cowls_excluded]
        print('Current data size: ', len(self.df))
        
        if self.record_file is not None:
            with open(self.record_file, 'r') as f:
                records = json.load(f)
                
            score_threshold = min(records['min_score'], records['min_score_cowls'])
            score_threshold = np.around(score_threshold, decimals=7)
            print('Score threshold: ', score_threshold)
            self.df = self.df[self.df['score'] >= score_threshold]
            
        print('Current data size after filtering: ', len(self.df))
        
        idx_selected_sl = self.df['selected_sl'] == 1
        idx_selected_non_sl = self.df['selected_non_sl'] == 1
        
        print('Selected SL sources: ', idx_selected_sl.sum())
        print('Selected non-SL sources: ', idx_selected_non_sl.sum())
        
        self.selected_sl_names += self.df[idx_selected_sl]['name'].tolist()
        self.selected_non_sl_names += self.df[idx_selected_non_sl]['name'].tolist()
    
        self.last_sl_count = len(self.selected_sl_names)
        
        df_names = self.df['name'].values.astype(str)
        
        data = np.load(self.data_path)
        self.latents = data['embeddings']
        self.galaxy_names = data['names']
        
        _, gal_idx, df_idx = np.intersect1d(self.galaxy_names, df_names, return_indices=True)
        self.latents = self.latents[gal_idx].astype(np.float32)
        self.galaxy_names = self.galaxy_names[gal_idx]
        self.df = self.df.iloc[df_idx].reset_index(drop=True)
        
        self.name_to_idx = {name: i for i, name in enumerate(self.galaxy_names.tolist())}
        
        self.latents = torch.from_numpy(self.latents).to(config.device, non_blocking=True)
        print('Latent shape: ', self.latents.shape)
        
    def initialize_ensembles(self):
        
        self.ensembles = []
        for i in range(self.maximum_ensemble_size):
            model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                 bayesian=False).to(config.device)
            self.ensembles.append(model)
        
        params, buffers = stack_module_state(self.ensembles)
        self.optimizer = optim.Adam(params.values(), lr=1e-3, weight_decay=1e-5, 
                                    fused=False)
        self.scaler = torch.amp.GradScaler(device=config.device)
        
        def fmodel(params, buffers, x):
            return functional_call(self.ensembles[0], (params, buffers), x)
        
        self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
        self.params = params
        self.buffers = buffers
        
        self.ensemble_size = len(self.ensembles)
        
        print(f'Initialized {self.ensemble_size} ensembles')
        
    def get_available_galaxies(self):
        
        # exclude selected SL and non-SL
        excluded_names = self.selected_sl_names + self.selected_non_sl_names
        
        excluded_names = np.array(excluded_names)
        
        self.available_names = np.setdiff1d(self.galaxy_names, excluded_names)
        
        return len(self.available_names)
    
    def add_selections(self, sl_names, non_sl_names):
        
        # Count only NEW selections (not duplicates)
        new_sl_count = len([name for name in sl_names if name not in self.selected_sl_names])
        new_non_sl_count = len([name for name in non_sl_names if name not in self.selected_non_sl_names])
        
        # Add new items while preserving order and track round added
        for name in sl_names:
            if name not in self.selected_sl_names:
                self.selected_sl_names.append(name)
                self.sample_round_added[name] = self.current_round
                self.sl_name_incremented.append(name)
        
        for name in non_sl_names:
            if name not in self.selected_non_sl_names:
                self.selected_non_sl_names.append(name)
                self.sample_round_added[name] = self.current_round
        
        # Increment submission counter by the number of NEW selections only
        self.total_submissions += new_sl_count + new_non_sl_count
        
        print(f'Added {new_sl_count} new SL and {new_non_sl_count} new non-SL selections')
        print(f'Total submissions: {self.total_submissions}')
    
    def train_model(self, epochs=None):
        
        increment = len(self.selected_sl_names) - self.last_sl_count
        self.num_increment_history.append(increment)
        self.last_sl_count = len(self.selected_sl_names)
        
        print('Increment SL sources: ', increment)
        
        if self.fix_ensembles == False and self.cold_start == False:
            self.add_ensemble()
        
        if self.fix_ensembles == True and self.cold_start == True:
            print('Resetting ensembles...')
            self.initialize_ensembles()
        
        sl_count = len(self.selected_sl_names)
        non_sl_count = len(self.selected_non_sl_names)
        total_count = sl_count + non_sl_count
        
        print(f"\n=== Training ensembles - Round {self.current_round} ===")
        print(f'Ensemble size: {self.ensemble_size}')
        print(f"SL sources: {sl_count}")
        print(f"Non-SL sources: {non_sl_count}")
        print(f"Total training samples: {total_count}")
        print(f"SL ratio: {sl_count/total_count:.2%}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Patience: {self.patience}")
        print("=" * 40)
        
        training_names = np.array(self.selected_sl_names + self.selected_non_sl_names)
        training_labels = np.array([1] * len(self.selected_sl_names) + [0] * len(self.selected_non_sl_names))
        training_labels = torch.from_numpy(training_labels).to(config.device, non_blocking=True)
        
        
        training_indices = np.array([self.name_to_idx[name] for name in training_names.tolist()])
        training_indices = torch.from_numpy(training_indices).to(config.device, non_blocking=True)
        
        training_latents = self.latents[training_indices]
        
        # Calculate class weights for imbalanced classes
        labels, counts = torch.unique(training_labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        
        print('Labels: ', labels.cpu().numpy())
        print('Counts: ', counts.cpu().numpy())
        print('Class weights: ', class_weights.cpu().numpy())
        
        class_sample_weights = class_weights[training_labels]
        # normalize to mean 1
        # class_sample_weights = class_sample_weights / class_sample_weights.mean()
        
        # Calculate recency weights (exponential decay)
        # recency_decay_alpha = config.recency_decay_alpha        
        # recency_weights = self.get_recency_weights(training_names, recency_decay_alpha)
        # recency_weights_tensor = torch.from_numpy(recency_weights).to(config.device, non_blocking=True)
        
        # # normalize to mean 1
        # recency_weights_tensor = recency_weights_tensor / recency_weights_tensor.mean()
        
        # # Combine class weights and recency weights (element-wise multiplication)
        # sample_weights = class_sample_weights * recency_weights_tensor
        
        sample_weights = class_sample_weights
        training_labels = training_labels.float()
        
        # if recency_decay_alpha != 0:
        #     print(f'Recency decay alpha: {recency_decay_alpha}')
        #     print(f'Recency weights - min: {recency_weights_tensor.min():.4f}, max: {recency_weights_tensor.max():.4f}, mean: {recency_weights_tensor.mean():.4f}')
        
        print(f'Class sample weights - min: {class_sample_weights.min():.4f}, max: {class_sample_weights.max():.4f}, mean: {class_sample_weights.mean():.4f}')
        print(f'Sample weights - min: {sample_weights.min():.4f}, max: {sample_weights.max():.4f}, mean: {sample_weights.mean():.4f}')
        
        dataloader = CudaDataLoader(
            training_latents,
            training_labels,
            self.batch_size,
            sample_weights,
        )
        
        num_batches = len(dataloader)
        
        print('Training ensembles...')
        
        start_time = time.time()
        
        training_losses = []
        best_loss = float('inf')
        best_params = copy.deepcopy(self.params)
        best_epoch = 0
        patience_counter = 0
        
        epoch = 0
        while True:
            
            train_epoch_loss = torch.tensor(0.0, device=config.device)
            
            for batch_latents, batch_labels in dataloader:
                
                self.optimizer.zero_grad()
                
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                
                    preds = self.predict_ensemble(self.params, self.buffers, batch_latents)
                    preds_flat = preds.reshape(-1, 1)
                    
                    labels_expanded = batch_labels.expand(len(self.ensembles), -1).reshape(-1, 1)
                    labels_expanded = labels_expanded.float()
                    
                    loss = self.criterion(preds_flat, labels_expanded)
                    
                    train_epoch_loss += loss.detach()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            # Single CPU sync per epoch
            avg_loss = (train_epoch_loss / num_batches).item()
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
                        print(f'Early stopping triggered at epoch {epoch} (patience: {self.patience})')
                        break
            
            epoch += 1
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Training time: {training_time:.2f} seconds')
        
        self.best_params = best_params
        
        self.model_trained = True
        self.round_save_path = os.path.join(f'{self.results_path}', 
                                            f'round_{self.current_round}')
        os.makedirs(self.round_save_path, exist_ok=True)
        
        print(f'Saving best model at epoch {best_epoch} with loss {best_loss:.4f}')
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
        
        os.makedirs(os.path.join(self.round_save_path, 
                                 'SL_candidates'), exist_ok=True)
        
        for name in self.sl_name_incremented:
            shutil.copy(os.path.join(self.images_path, name + '.jpg'), 
                        os.path.join(self.round_save_path, 'SL_candidates', name + '.jpg'))
        
        # re-initialize sl_name_incremented for next round
        self.sl_name_incremented = []
        
        return {
            'success': True,
            'round': self.current_round,
            'epochs_trained': epoch + 1,  # Total epochs trained (epoch is 0-indexed)
            'best_epoch': best_epoch,
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
        
        # All latent vectors are already stored on the device in the same
        # order as self.galaxy_names, so we can use them directly without
        # rebuilding indices on the CPU.
        all_names = self.galaxy_names
        all_latents = self.latents
        
        dataloader = CudaDataLoader(
            all_latents,
            y_tensor=None,  # Prediction mode - only x_tensor needed
            batch_size=self.batch_size,
        )
        
        # Accumulate predictions on GPU and move to CPU once at the end
        scores_gpu = []
        for batch_latents in dataloader:
            
            with torch.no_grad():
                
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    preds_logits = self.predict_ensemble(self.best_params, self.buffers, batch_latents)
                    # Convert logits to probabilities for scoring
                    preds = torch.sigmoid(preds_logits)
                    
            scores_gpu.append(preds)  # (num_ensembles, batch_size)
            
        end_time = time.time()
        updating_time = end_time - start_time
        print(f'Updating scores time: {updating_time:.2f} seconds')
                
        scores = torch.cat(scores_gpu, dim=1).cpu().numpy()  # (num_ensembles, num_samples)
        mean_scores = np.mean(scores, axis=0)
        uncertainties = np.std(scores, axis=0)
        
        self.scores = {name: score for name, score in zip(all_names, mean_scores)}
        self.uncertainties = {name: uncertainty for name, uncertainty in zip(all_names, uncertainties)}
        
        dataframe = self.df.copy()
        dataframe.drop(
            columns=['score', 'uncertainty', 'selected_sl', 'selected_non_sl'],
            inplace=True,
            errors='ignore',
        )
        
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
            'model_resetted': self.model_resetted if hasattr(self, 'model_resetted') else False,
            'num_ensembles': self.ensemble_size,
            'dividing_threshold': self.dividing_threshold,
            'min_score': self.stats['min_score'],
            'num_over_min_score': self.stats['num_over_min_score'],
            'min_score_cowls': self.stats['min_score_cowls'],
            'num_over_min_score_cowls': self.stats['num_over_min_score_cowls'],
            'num_lower_max_non_sl_score': self.stats.get('num_lower_max_non_sl_score', 0),
            'purity': len(self.selected_sl_names) / self.stats['num_over_min_score_cowls'] if self.stats['num_over_min_score_cowls'] > 0 else 0.0,
            'sl_name_incremented': self.sl_name_incremented,
            'sl_names': self.selected_sl_names,
            'non_sl_names': self.selected_non_sl_names,
            'sample_round_added': self.sample_round_added,  # Save round tracking
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