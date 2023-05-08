# DL-SIM
High-fidelity reconstruction of structured illumination microscopy by an amplitude-phase channel attention network with multitemporal information
# Dependencies
  * Python 3.7.10
  * PyTorch 1.9.0
  * numpy 1.18.5
  * imageio 2.9.0
  * matplotlib 3.4.2
  * scikit-image 0.18.2
  * torchvision 0.10.0

# Usage
  ## training
    # Some hyper-paramters in options.py, you can edit them!
    # Before training, you need to prepare the training dataset and place it in the dataset directory.
    # Then, run train.py.
  
  ## testing
    # If you want to use the pre-trained model with APCAN w/o time, then you need to set the opt.model to 'apcan_actin_1' for Actin, 'apcan_er_1' for ER in test.py.
    # If you want to use the pre-trained model with APCAN w time, then you need to set the opt.model to 'apcan_actin_3' for Actin, 'apcan_er_3' for ER in test.py.
    # Then, you need to change the input path and output path in test.py.
    # Finally, run test.py.
    
# T-SIM Dataset
  ## The T-SIM is a multi-time-point structured illumination microscopy super-resolution dataset containing a total of two sub-datasets, an Endoplasmic Reticulum dataset of 10 time points and a F-actin dataset of 20 time points.The T-SIM dataset can be downloaded through the following link：
    1、https://figshare.com/articles/dataset/F-actin-Cell001-Cell004/22776098
    2、https://figshare.com/articles/dataset/F-actin-Cell005-Cell008/22776116
    3、https://figshare.com/articles/dataset/F-actin-Cell009-Cell012/22777142
    4、https://figshare.com/articles/dataset/F-actin-Cell013-Cell016/22777154
    5、https://figshare.com/articles/dataset/Endoplasmic_Reticulum/22774850
