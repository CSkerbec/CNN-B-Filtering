import numpy as np
import tensorflow as tf
from tensorflow.keras import *
import matplotlib.pyplot as plt
import sys

#generate Q/U maps for each value of ell
def generate_QU(ell,Dl_EE,Dl_BB,npix=512, map_size=20):
    #converting power spectra data (Dl) data to Cl
    Cl_EE = Dl_EE / ell / (ell + 1.0) * (2.0 * np.pi)
    Cl_BB = Dl_BB / ell / (ell + 1.0) * (2.0 * np.pi)
    
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #create input npix by npix gaussian random fields
    random_array_E = np.fft.fft2(np.random.normal(0,1,(npix,npix)))
    random_array_B = np.fft.fft2(np.random.normal(0,1,(npix,npix)))

    #interpolate B and multiply by the randomized array
    E_FS = np.interp(l, ell, np.sqrt(Cl_EE), right=0)*random_array_E
    B_FS = np.interp(l, ell, np.sqrt(Cl_BB), right=0)*random_array_B
    
    #spin 2
    Q_FS = E_FS*np.cos(2*phi) - B_FS*np.sin(2*phi)
    U_FS = E_FS*np.sin(2*phi) + B_FS*np.cos(2*phi)
    
    #Convert T, Q, U to real space using inverse fourier transform
    #ifftshift will shift zero-frequency component to center of spectrum
    Q_RS = np.fft.ifft2(np.fft.ifftshift(Q_FS))
    Q_map = np.real(Q_RS)*npix*np.pi 
    
    U_RS = np.fft.ifft2(np.fft.ifftshift(U_FS))
    U_map = np.real(U_RS)*npix*np.pi 

    return Q_map,U_map

#this creates one map, now loop the function to create more maps
def create_maps(ell,Dl_EE,Dl_BB,nrlz,npix=512, map_size=20):
    CMB_maps=np.zeros((2,npix,npix,nrlz)) #hard code the 2?
    for i in range(nrlz):
        Q,U=generate_QU(ell,Dl_EE,Dl_BB)
        CMB_maps[0,:,:,i]=Q
        CMB_maps[1,:,:,i]=U
    return CMB_maps

#now mask the maps, this also switches axes to (batch_size,npix,npix,Q/U)
def masking(CMB_maps,mask,noise):
    masked_maps = np.transpose(np.zeros(CMB_maps.shape),(3,1,2,0))
    for i in range(CMB_maps.shape[-1]):
        masked_maps[i,:,:,:]=np.transpose((CMB_maps[:,:,:,i]*mask)+noise,(1,2,0))
    return masked_maps

#create mask in shape of masked QU maps
def mask_layer(masked_maps,smooth_mask):
    smoothed_mask = np.zeros(masked_maps.shape)
    for i in range(masked_maps.shape[0]):
        for j in range(masked_maps.shape[-1]):
            smoothed_mask[i,:,:,j]=smooth_mask
    return smoothed_mask
            

def standardize(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


#now define the power spectrum function
def power_spectra(QU, npix=512, map_size=20,bin_size=801,bin_interval=40):
    #Transforming from real space(RS) to Fourier space(FS) using 2-d Fourier Transform,
    #A factor of(np.pi*(npix**2)) is needed
    Q = QU[:,:,0]
    U = QU[:,:,1]
    QN = np.fft.fftshift(np.fft.fft2(Q))/(np.pi*(npix**2))
    UN = np.fft.fftshift(np.fft.fft2(U))/(np.pi*(npix**2))
    
    #define Fourier plane coordinates
    dl = 2.0*np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix),np.arange(-npix/2,npix/2)*dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2+ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0
    
    #Inverted rotation matrix 
    #line segments of E and B mode are symmetric over pi radians
    E = QN*np.cos(2*phi) + UN*np.sin(2*phi)
    B =-QN*np.sin(2*phi) + UN*np.cos(2*phi)
    
    #apply conjugate and doing operation to convert Cl to Dl
    EE=np.real(np.conj(E)*E)*(l*(l+1))/(2*np.pi)
    BB=np.real(np.conj(B)*B)*(l*(l+1))/(2*np.pi)
    EB=np.real(np.conj(E)*B)*(l*(l+1))/(2*np.pi)
    
    #define bin.. goes from {0,1,2,...,bin_size} in steps of bin_interval
    bin_edge = np.arange(0, bin_size, bin_interval)
    
    #Create list of zeros filled in with mean values calculated in for loop
    DlEE = np.zeros(len(bin_edge)-1)
    DlBB = np.zeros(len(bin_edge)-1)
    DlEB = np.zeros(len(bin_edge)-1)
    
    #finding mean at values of l for all types of spectra
    for i in range(len(DlEE)):
        DlEE[i]=np.mean(EE[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlBB[i]=np.mean(BB[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlEB[i]=np.mean(EB[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        
    return DlEE, DlBB, DlEB

#run the power spectrum function in a loop
def power_loop(QU_input):
    DlEE = np.zeros((20,QU_input.shape[0]))#hard coded 20
    DlBB = np.zeros((20,QU_input.shape[0]))
    DlEB = np.zeros((20,QU_input.shape[0]))
    for i in range(QU_input.shape[0]):
        DlEE[:,i], DlBB[:,i], DlEB[:,i]=power_spectra(QU_input[i,:,:,:])
    return DlEE,DlBB,DlEB

if __name__=="__main__":
    #loop over specified ell values to run on (EE/BB spectrum)
    e100 = models.load_model("../trained_models/02-23-21/5layer/models/100.hdf5")
    mask = np.load('../mapdata/original_data/mask_v0.npy')
    noise=(np.random.randn(512,512)/(np.sqrt(mask+10**-4)*20))    
    
    start_ell = np.int32(sys.argv[1]) #starting ell 
    end_ell = np.int32(sys.argv[2]) #ending ell 
    step_ell = np.int32(sys.argv[3]) #stepsize of ell
    nrlz = np.int32(sys.argv[4])
    polarization = sys.argv[5] #EE/BB create a if statement for EE/BB 
    
    input_ell = np.arange(start_ell,end_ell,step_ell) 
    ell = np.arange(1,1500)
    
    DlEE = np.zeros((len(input_ell),20,nrlz)) #input_ell value, output data, number of realizations
    DlBB = np.zeros((len(input_ell),20,nrlz))
    DlEB = np.zeros((len(input_ell),20,nrlz))
    
    for i in range(len(input_ell)): #create spectra for ell values (0 at all other than specified ell)
        input_EE = np.zeros(ell.shape)
        input_BB = np.zeros(ell.shape)
        
        if polarization == "EE": ##need to create arrays with 0's everywhere other than specified ell value
            input_EE[(ell>=input_ell[i])&(ell<input_ell[i]+step_ell)] = 1.0
        elif polarization == "BB":
            input_BB[(ell>=input_ell[i])&(ell<input_ell[i]+step_ell)] = 1.0                                         
        
        ##generate Q/U maps##
        QU = create_maps(ell,input_EE,input_BB,nrlz,map_size=10) #loop over ell values #output will be 2 Q/U maps
        masked_maps = masking(QU,mask>0,noise) 

        masked_layer = mask_layer(masked_maps,mask) #creates last two layers of input data 
        
        #concatenate input maps with smoothed mask
        input_QU=np.concatenate((masked_maps,masked_layer),axis=3)
        padded_QU=np.pad(input_QU,((0,0),(64,64),(64,64),(0,0)),mode="constant")
        
        #standardize the data
        model_input=np.zeros(padded_QU.shape)

        model_input[:,:,:,2:4]=padded_QU[:,:,:,2:4]
        model_input[:,:,:,0:2]=standardize(padded_QU[:,:,:,0:2],axis=(1,2))

        #run through the model to get "predicted" Q/U
        output_model = e100.predict(model_input,batch_size=1)
        print(output_model.shape)
        ##calculate EE/BB power spectra##
        DlEE[i,:,:],DlBB[i,:,:],DlEB[i,:,:] = power_loop(output_model)
        
    #save all the power spectra to an file
    np.savez("../mapdata/bpwf/bpwf_Nmodel_{:03d}_{}.npz".format(start_ell,polarization),input_ell,DlEE,DlBB,DlEB)