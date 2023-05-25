## Flexible Variable-Rate Image Feature Compression for Edge-Cloud Systems

## Required packages
numpy\
opencv-python\
pytorch >=1.12\
tqdm\
compressai==1.2.2

To build an exact new conda environment from the ```spec-file.txt``` use the command:
```
conda create --name <name_of_environment> --file spec-file.txt
```

To install required packages into an existing environment using ```spec-file.txt``` use the command:
```
conda install --name <name_of_environment> --file spec-file.txt
```

## Viewing results from the paper
Experiment 1 (Comparison between variable-rate and fixed-rate images):
```
python plot-results.py --results_dir results_scaled_hyperprior_factorized_prior
```

Experiment 2 (Effect of each component of Conditional Convolution block):
```
python plot-results.py --results_dir results_ablation
```

Experiment 3 (Comparison of proposed methods with JPEG and WebP):
```
python plot-results.py --results_dir results_baseline
```

The plots for each of the three experiments can be found in the images directory.
The model definitions are in the file ```models/vae.py```.
The training script is ```train.py```.
The evaluation script for variable rate models is ```evaluate.py```.
the evaluation script for fixed rate models is ```evaluate_fixed.py```.
The evaluation script for JPEG and WebP is ```jpeg.py```.
```utils.py``` contains helper functions for training and data loading.

## Training
To train VARIABLE-RATE models on one GPU use the command:
```
python train.py --model <factorized_prior/scale_hyperprior> --wbmode disabled --epochs <num_epochs_to_train>
```

To train FIXED-RATE models on one GPU use the command:
```
python train.py --model <factorized_prior_fixed/scale_hyperprior_fixed> --wbmode disabled --epochs <num_epochs_to_train>
```

To train on multiple GPUs (4 in the example command below) use the command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 train.py --model <model_name> --wbmode disabled --epochs <num_epochs_to_train> --ddp_find
```

If you want to track traing progress with wandb create an account in wandb and set the argument ```--wbmode online```.
A new project by the name of "learned-image-compression" will be created where you can monitor training progress.
To create an account in wandb see: "https://docs.wandb.ai/quickstart".
Trained models can be found in ```runs/learned-image-compression```.

## Evaluation
To evaluate a trained model, place the model weights in the directory ```checkpoints```.

To evaluate VARIABLE-RATE models run the command:
```
python evaluate.py --models <factorized_prior/scale_hyperprior> --checkpoint <checkpoint_name(default:scale_hyperprior_2)>
```

To evaluate FIXED-RATE models run the command:
```
python evaluate.py --models <factorized_prior_fixed/scale_hyperprior_fixed>
```

To evaluate JPEG:
```
python jpeg.py
```
