#from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import io as sio 
import os 
import math
from sys import argv
total_movie_frames = 7204



def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def experiment_perFrame_nikitas(qpnum=51, 
        model_name='Single', NR_path='./noisy_imgs', 
        signal_path='./signal_imgs', results_path='./'):

    signal_dir = os.listdir(signal_path)
    movie_name = signal_dir[0].split('_')[0]
    
    NR_folder_file_count = len([name for name in os.listdir(NR_path) if True])
    signal_folder_file_count = len([name for name in signal_dir if True])
    
    #assert  signal_folder_file_count == NR_folder_file_count
    
    
    # results_file_name = os.path.join (results_path, (movie_name+'_'+model_name + '.mat'))
    results_file_name = os.path.join  (results_path, (model_name+'.mat'))
    
    if os.path.exists(results_file_name):
        mat_contents = sio.loadmat(results_file_name)
        results_NR = mat_contents['results_NR']
        print (results_NR)
        if results_NR is None:
            results_NR = np.zeros((total_movie_frames,1))
    else:
        results_NR = np.zeros((total_movie_frames,1))
        mat_contents = {}
    
    for (file_idx, file_name) in enumerate(os.listdir(NR_path)):
        signal_file = os.path.join( signal_path,   file_name)
        NR_file = os.path.join( NR_path,   file_name)
        
        signal_image = cv2.imread(signal_file)
        NR_image = cv2.imread(NR_file)
        
        image_psnr = psnr(signal_image,NR_image)
        results_NR [file_idx] = image_psnr;
        print (str(image_psnr) + ' for ' + str(file_idx))
        
    mat_contents['results_NR'] = results_NR
    sio.savemat (results_file_name, mat_contents);
        
        
            
if __name__ == "__main__":
    
    model_name = argv[1]
    NR_path = argv[2]
    results_path=argv[3]
    RAW_path = argv[4]
    '''
    bic_path = argv[3]
    signal_path=argv[4]
    trimmer = int(argv[6])
    qpnum = int(argv[7])
    '''
    
    
    
    experiment_perFrame_nikitas(
        model_name=model_name, NR_path=NR_path, 
        signal_path=RAW_path, results_path=results_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
