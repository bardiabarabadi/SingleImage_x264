# Keras-ViedoEnhance

### Geneate Dataset
To generate a dataset you need to have RAW 540p videos. Using `GenDataset.bash` you can convert the videos to frames and generate a dataset.

    mkdir Temp/Dir
    bash GenDataset.bash Dir/to/videos/ QP Dir/to/destination/ Temp/Dir

Make sure you create 'Temp/Dir/' before running the script. Also, in your videos directory, there should be two folders, **train** and **test**, containing **.mov** files. A the destination dataset will have the same organization.

### Train a Model
Train a model using an exisiting dataset by using `train.py`. The parameters are stated below:

    python train.py 
                                --LR_dir                    [Directory containing RAW, 540p images (QP=0)]
                                --NR_dir                    [Directory containing compressed, 540p images (QP!=0)]
                                --model_save_dir    [Directory to save model weights, file name: "model_best_weights.h5"]
                
### Test a Model (Generate Enhanced Images)

Test a model on a set of compressed images by using `test.py`. Required parameters are shown below:

    python test.py
                                -inr            [Directory containing compressed images, QP!=0]
                                -o               [Output directory to save enhanced images]
                                -m              [The model file to use for test (.../model_best_weights.h5)]
                      
### Evaluate Enhancement (PSNR Calculation)

To compare the enhanced results (or the not-enhanced compressed frames) with RAW frames and calculate PSNR, `PSNR_gen.py` can be used.

    python PSNR_gen.py
                                SAVE_FILE         [File name to save PSNR results (.mat)]
                                NOISE_PATH      [Directory containing noisy images]
                                SAVE_PATH       [Directory to save PSNR results (.mat)]
                                SIGNAL_PATH    [Directory containing signal (noNoise) images]
                                
                        