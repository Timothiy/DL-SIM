# DL-SIM
High-fidelity reconstruction of structured illumination microscopy by an amplitude-phase channel attention network with multitemporal information
# Dependencies
  

# Usage
  ## training
    # some hyper-paramters in options.py, you can edit them!
    # Before training, you need to prepare the training dataset and place it in the dataset directory.
    # Then, run train.py.
  
  ## testing
    If you want to use the pre-trained model with APCAN w/o time, then you need to set the opt.model to 'apcan_actin_1' for Actin,
    'apcan_er_1' for ER in test.py.
    If you want to use the pre-trained model with APCAN w time, then you need to set the opt.model to 'apcan_actin_3' for Actin,
    'apcan_er_3' for ER in test.py.
    Then, you need to change the input path and output path in test.py.
    Finally, run test.py.
