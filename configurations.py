import argparse
import json
import os
from types import SimpleNamespace

def str_to_bool(value):
    """Convert string to boolean value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

# Parse command line arguments
parser = argparse.ArgumentParser()

# paths
parser.add_argument('--data_path', type=str, 
                    default='../final_ds/embeddings.npz', 
                    help='Path to the embeddings file')
parser.add_argument('--images_path', type=str, 
                    default='../final_ds/JWST_images', 
                    help='Path to the images directory')
parser.add_argument('--dataframe_path', type=str, 
                    default='../final_ds/JWST_SL_discovery_catalog_24.5.csv', 
                    help='Path to the dataframe file')
parser.add_argument('--results_path', type=str, default='results',
                    help='Path to the results directory')
parser.add_argument('--score_file', type=str, default=None, 
                    help='Path to the score file for the second stage')
parser.add_argument('--record_file', type=str, default=None, 
                    help='Path to the record file for the second stage')

# data parameters
parser.add_argument('--latents_scaled', type=str_to_bool,
                    default=False,
                    help='Whether to scale the latents')
parser.add_argument('--mag_limit', type=float,
                    default=None,
                    help='Magnitude limit')
parser.add_argument('--score_limit', type=str,
                    default=None, 
                    help='COWLS score limit')
parser.add_argument('--spring_dataset', type=str_to_bool,
                    default=False,
                    help='Whether to use the spring dataset')
parser.add_argument('--spring_data_path', type=str,
                    default='../spring_datasets/embeddings_spring.npz',
                    help='Path to the spring dataset')
parser.add_argument('--use_focal_loss', type=str_to_bool,
                    default=False,
                    help='Whether to use focal loss')


# model parameters
parser.add_argument('--embedding_size', type=int,
                    default=256,
                    help='Embedding size')
parser.add_argument('--maximum_ensemble_size', type=int,
                    default=5,
                    help='Maximum ensemble size')
parser.add_argument('--patience', type=int, 
                    default=40,
                    help='Patience')
parser.add_argument('--warmup_epochs', type=int,
                    default=300,
                    help='Warmup epochs')
parser.add_argument('--batch_size', type=int,
                    default=2048,
                    help='Batch size')
parser.add_argument('--norm_method', type=str,
                    default='layer',
                    help='Normalization method')

# other parameters
parser.add_argument('--supplement_ratio', type=float, 
                    default=0.2,
                    help='Supplement ratio')
parser.add_argument('--supplement_method', type=str,
                    default='uncertainty',
                    help='Supplement method')
parser.add_argument('--num_submission_train', type=int,
                    default=100,
                    help='Number of submission train')
parser.add_argument('--random_seed', type=int, default=1234, 
                    help='Random seed')
parser.add_argument('--fix_ensembles', type=str_to_bool,
                    default=False,
                    help='Whether to fix the ensembles')
parser.add_argument('--cold_start', type=str_to_bool,
                    default=False,
                    help='Whether to use cold start')
parser.add_argument('--recency_decay_alpha', type=float,
                    default=0.1,
                    help='Exponential decay rate for recency weighting (higher = faster decay, more focus on recent samples)')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')

# resume from checkpoint
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='Host to bind (e.g. 0.0.0.0 for all interfaces, or 10.0.10.80 for a specific IP)')
parser.add_argument('--port', type=int, default=6543,
                    help='Port number')
parser.add_argument('--checkpoint_round', type=int, default=None,
                    help='Checkpoint round')
parser.add_argument('--processing_stage', type=str, default='first', 
                    help='Processing stage')

args = parser.parse_args()

config = args

os.makedirs(config.results_path, exist_ok=True)

# if use checkpoint, load existing configs
if config.checkpoint_round is not None:
    
    print(f'Loading configs')
    with open(f'{config.results_path}/config.json', 'r') as f:
        saved_config = json.load(f)
    
    # ensure host exists (for configs saved before --host was added)
    if 'host' not in saved_config:
        saved_config['host'] = getattr(config, 'host', '0.0.0.0')
    # modify configs by input args
    for key, value in vars(config).items():
        if key in saved_config and saved_config[key] != value:
            print(f'Modify {key}: {saved_config[key]} -> {value}')
            saved_config[key] = value
    
    # final config from saved configs and input args
    config = SimpleNamespace(**saved_config)
    

# save final configs
with open(f'{config.results_path}/config.json', 'w') as f:
    json.dump(vars(config), f, indent=4)
