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
