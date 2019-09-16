# Adversarial rotations and translations for CIFAR10

This repository contains code to train and evaluate CIFAR10 models against
adversarially chosen rotations and translations (code for ImageNet at https://github.com/MadryLab/spatial-pytorch). It can be used to reproduce the
main experiments of:

**Exploring the Landscape of Spatial Robustness**<br>
*Logan Engstrom\*, Brandon Tran\*, Dimitris Tsipras\*, Ludwig Schmidt, Aleksander
   Mądry*<br>
ICML 2019<br>
http://arxiv.org/abs/1712.02779

The main scipts to run are `train.py` and `eval.py`, which will train and
evaluate a model respectively. Options are all included in `config.json`
annotated below.

```
{
  "model": {
      "output_dir": "output/test",
      # padding mode, passed directly to tf.pad
      "pad_mode": "constant", 
      "filters": [16, 16, 32, 64],
      # size of image fed to classifier,set to 64 for black-canvas setting (no
      # information loss during rotation and translation)
      "pad_size": 32
  },

  "training": {
      "tf_random_seed": 557212,
      "np_random_seed": 993101,
      "max_num_training_steps": 80000,
      "num_output_steps": 100,
      "num_summary_steps": 100,
      "num_eval_steps": 500,
      "num_checkpoint_steps": 500,
      "batch_size": 128,
      "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
      "momentum": 0.9,
      "weight_decay": 0.0002,
      # interleaves evaluation steps during training, useful for single GPU runs
      "eval_during_training": true,
      # include Linf and spatial attacks during training
      "adversarial_training": false,
      # use random left-right flip (see note below)
      "data_augmentation": true
  },

  "eval": {
      "num_eval_examples": 10000,
      "batch_size": 128,
      # useful for quickly computing standard accuracy if set to false
      "adversarial_eval": true
  },

  "attack": {
      # perform Linf-bounded PGD attack
      "use_linf": false,
      # perform adversarial rotations and translations
      "use_spatial": true,

      # parameters for PGD attacks
      "loss_function": "xent", # can also be set to "cw" for Carlini-Wagner
      "epsilon": 8.0,
      "num_steps": 5,
      "step_size": 2.0,
      "random_start": false,

      # parameters for spatial attack
      # can either be chosen using a few random tries or exhaustive grid search
      "spatial_method": "random", # or "grid"
      "spatial_limits": [3, 3, 30], # trans_x pix, trans_y pix, rotation degrees
      "random_tries": 10, # if method is random choose the worst of x tries
      "grid_granularity": [5, 5, 31] # controls how many points are in the grid
  },

  "data": { "data_path": "/scratch/datasets/cifar10" }
}
```

Data augmentation only included random left-right flips. Standard CIFAR10
augmentation (+-2 pixel crops) can be achieved by setting
`adversarial_training: true`, `spatial_method: random`, `random_tries: 1`,
`spatial_limits: [2, 2, 0]`.
