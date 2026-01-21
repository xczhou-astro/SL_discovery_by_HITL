import argparse
import json
import os

# Parse command line arguments
parser = argparse.ArgumentParser()

# paths
parser.add_argument('--data_path', type=str, 
                    default='../datasets/embeddings.npz', 
                    help='Path to the embeddings file')
parser.add_argument('--images_path', type=str, 
                    default='../datasets/JWST_images', 
                    help='Path to the images directory')
parser.add_argument('--dataframe_path', type=str, 
                    default='../datasets/JWST_SL_discovery_catalog.csv', 
                    help='Path to the dataframe file')
parser.add_argument('--results_path', type=str, default='results',
                    help='Path to the results directory')

# data parameters
parser.add_argument('--latents_scaled', type=bool,
                    default=False,
                    help='Whether to scale the latents')
parser.add_argument('--filter', type=bool,
                    default=True,
                    help='Whether to filter the data')
parser.add_argument('--mag_limit', type=float,
                    default=24.6,
                    help='Magnitude limit')


# model parameters
parser.add_argument('--embedding_size', type=int,
                    default=384,
                    help='Embedding size')
parser.add_argument('--maximum_ensemble_size', type=int,
                    default=5,
                    help='Maximum ensemble size')
parser.add_argument('--patience', type=int, 
                    default=20,
                    help='Patience')
parser.add_argument('--epochs', type=int,
                    default=500,
                    help='Epochs')
parser.add_argument('--warmup_epochs', type=int,
                    default=300,
                    help='Warmup epochs')
parser.add_argument('--batch_size', type=int,
                    default=1024,
                    help='Batch size')
parser.add_argument('--norm_method', type=str,
                    default='layer',
                    help='Normalization method')

# other parameters
parser.add_argument('--supplement_ratio', type=float, 
                    default=0.3,
                    help='Supplement ratio')
parser.add_argument('--supplement_method', type=str,
                    default='uncertainty',
                    help='Supplement method')
parser.add_argument('--num_submission_train', type=int,
                    default=100,
                    help='Number of submission train')
parser.add_argument('--random_seed', type=int, default=1234, 
                    help='Random seed')
parser.add_argument('--fix_ensembles', type=bool,
                    default=True,
                    help='Whether to fix the ensembles')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')

# checkpoint
parser.add_argument('--checkpoint_round', type=int, default=None,
                    help='Checkpoint round')

args = parser.parse_args()

config = args

os.makedirs(config.results_path, exist_ok=True)
with open(f'{config.results_path}/config.json', 'w') as f:
    json.dump(vars(config), f, indent=4)