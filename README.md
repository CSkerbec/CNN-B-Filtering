# CNN-B-Filtering
CMB polarization filtering with convolutional neural network. Inputs used Q/U binary masked maps (100,640,640,4) --- (# maps, npix, npix, Q/U + 2 masks)

/functions ---
CMB_functions.py has functions to create randomized Q/U maps from power spectra data and calculate power spectra from polarization maps. 

/map_data ---
Original data from CAMB can be transformed into QU maps using generate_CMB_maps under functions/CMB_functions. Simulated_Data.ipynb shows this. 

/model ---
5 layer model using encoder/decoder paths can be found under model/model.py file. 5 layer model is final model in the file. 

To train and graph the model go to model/5layer_graphs.ipynb, the trained model (using binary masked input maps and unmasked target maps) can be found under trained_model. 

/notebooks ---
Bandpower window function notebook. 


