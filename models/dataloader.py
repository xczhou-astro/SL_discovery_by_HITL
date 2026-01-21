import torch


class CudaDataLoader:
    def __init__(self, x_tensor, y_tensor=None, batch_size=None, sampler_weights=None, shuffle=True):
        """
        CudaDataLoader for efficient GPU-based data loading.
        
        Args:
            x_tensor: Input tensor (required)
            y_tensor: Target tensor (optional, for training). If None, only yields x_tensor (for prediction)
            batch_size: Batch size (required if y_tensor is provided, optional if y_tensor is None)
            sampler_weights: Sample weights for weighted sampling (optional)
            shuffle: Whether to shuffle data (default: True, ignored if y_tensor is None)
        """
        self.x = x_tensor
        self.y = y_tensor
        self.batch_size = batch_size if batch_size is not None else x_tensor.size(0)
        self.weights = sampler_weights
        self.shuffle = shuffle
        self.n_samples = x_tensor.size(0)
        self.prediction_mode = y_tensor is None

    def __iter__(self):
        # PREDICTION MODE: Only yield x_tensor (no labels)
        if self.prediction_mode:
            for i in range(0, self.n_samples, self.batch_size):
                yield self.x[i : i + self.batch_size]
            return
        
        # TRAINING MODE: Yield (x_tensor, y_tensor) tuples
        # STRATEGY A: Weighted Sampling (If weights provided)
        if self.weights is not None:
            # We determine how many batches to run. 
            # Usually we run len(data) / batch_size steps, 
            # even though we are oversampling the minority class.
            num_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
            
            for _ in range(num_batches):
                # torch.multinomial is the GPU equivalent of WeightedRandomSampler
                # replacement=True is crucial for oversampling minority classes
                indices = torch.multinomial(self.weights, self.batch_size, replacement=True)
                
                yield self.x[indices], self.y[indices]
                
        # STRATEGY B: Standard Shuffle (If no weights)
        elif self.shuffle:
            perm = torch.randperm(self.n_samples, device=self.x.device)
            for i in range(0, self.n_samples, self.batch_size):
                indices = perm[i : i + self.batch_size]
                yield self.x[indices], self.y[indices]
                
        # STRATEGY C: Sequential (Validation/Test)
        else:
            for i in range(0, self.n_samples, self.batch_size):
                yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size