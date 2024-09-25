python3 evaluate_models.py --uncertainty_method=ENS --save_model=True --use_wandb=True --ensemble_size=3 --force_weight=1
python3 evaluate_models.py --uncertainty_method=ENS --save_model=True --use_wandb=True --ensemble_size=6 --force_weight=1
python3 evaluate_models.py --uncertainty_method=ENS --save_model=True --use_wandb=True --ensemble_size=9 --force_weight=1
python3 evaluate_models.py --uncertainty_method=MVE --save_model=True --use_wandb=False --force_weight=1
python3 evaluate_models.py --uncertainty_method=EVI --save_model=True --use_wandb=True --force_weight=1

