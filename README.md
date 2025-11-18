## Data preparation
Step 1: Prepare free-form surfaces that will be used in the training process.

Step 2: Use the method described in sub-section "2.2 Dataset preparation" to generate corresponding training data.

Step 3: Put the curvature and height cloud maps under the directory './datasets/GridData/train_A' and './datasets/GridData/train_H', respectively, and then place the free-form grid structure images under the folder './datasets/GridData/train_B'.

Step 4: The corresponding text data will be generated based on the data within the folder train_B, and the text data will be stored within './datasets/GridData/train_C'.

## Train process
Use the command line below to start the training under the folder GridGAN, namely the folder where train.py file exists:
```bash
!# all parameters, except the parameters below, will be set as default and are stored under folder './options'  
python train.py --name GridGAN-TXT --tf_log --label_nc 0 --lambda_feat 15 --no_flip --dataroot ./datasets/GridData-TXT --n_downsample_global 4 --n_blocks_global 9 --lr 0.00005 --niter 50 --niter_decay 50
```

## Test process
After the train of the model has finished, run the command below to check the test result.
```bash
python test.py --name GridGAN-TXT --dataroot ./datasets/GridData-TXT --n_downsample_global 4 --n_blocks_global 9 --which_epoch latest
```
Note: Only the core file is shown here. The mutual call relationships between the core file and other files are identical to those in GridGAN. For further details, refer to https://github.com/GeneralHou/OA_GridGAN.
