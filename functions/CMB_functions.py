from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#mask = np.load('../mapdata/original_data/mask_v0.npy')


def standardize(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

    
def plot_corr(correlated,model_maximum=100,Title="", polarization="B"):
    plt.figure()
    epoch_interval=int(model_maximum/correlated.shape[0])
    multiple=int(model_maximum/epoch_interval)
    labels = []
    for i in range (1,multiple+1): 
        labels.append("e" + str(i*epoch_interval))
    for i in range(correlated.shape[0]):
        plt.plot(np.arange(0,1460,40)+20,correlated[i,:],label=labels[i])
    plt.title(polarization + " Correlation: " + Title)
    plt.xlabel('Multipole moment, $\ell$')
    plt.legend()
    plt.show()

#here we are taking the singular map generation code that originally worked and looping instead when function is called.
#The problem here is the Cholesky decomp. will be calculated in each iteration of the loop, this will result in extra computational time.
#However, when called in a loop, maps with correct scales are produced. 

def generate_CMB_maps(ell,Dl_TT,Dl_EE,Dl_BB,Dl_TE,npix=512, map_size=20):
    """
    generate_CMB_maps

    Create a random simulation of CMB temperature and polarization fluctuations from 
    input power spectra where T and E have partial correlation.

    Inputs
    ------
    ell: array
        Input "x" values.
    Dl_TT, Dl_EE, Dl_BB, Dl_TE : array
        CMB angular power spectrum in the form of D_ell 
        
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Outputs
    -------
    T_map : array, shape=(npix,npix)
        Temperature map generated from angular power spectrum
    Q_map : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U_map : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    
    """
        
    #converting power spectra data (Dl) data to Cl
    Cl_TT = Dl_TT / ell / (ell + 1.0) * (2.0 * np.pi)
    Cl_EE = Dl_EE / ell / (ell + 1.0) * (2.0 * np.pi)
    Cl_BB = Dl_BB / ell / (ell + 1.0) * (2.0 * np.pi)
    Cl_TE = Dl_TE / ell / (ell + 1.0) * (2.0 * np.pi)
    
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #create input npix by npix gaussian random fields
    #X and Y must be partially correlated and will be used for T & E partially correlated fields
    random_array_X = np.fft.fft2(np.random.normal(0,1,(npix,npix)))
    random_array_Y = np.fft.fft2(np.random.normal(0,1,(npix,npix)))
    random_array_B = np.fft.fft2(np.random.normal(0,1,(npix,npix)))
    
    #inputting arrays that match input npix that have complex components, every component is 0
    T_FS = np.zeros(random_array_X.shape,dtype=np.complex)
    E_FS = np.zeros(random_array_X.shape,dtype=np.complex)
    
    #setting 2x2 matrix with 0 components
    Cov_TE = np.matrix(np.zeros((2,2)))
    
    #need to generate T & E (correlated) Fourier maps
    #for loop to do calculations at every pixel
    #output 512x512 T_FS and E_FS matrices filled in with elements calculated within loop
    for i in range (len(lx)):
        for j in range(len(ly)):
            #Create 512, 2x2 matrices [[TT,TE],[TE,EE]] 
            #Interpolate TT,EE,TE spectra to this point in Fourier grid
            Cov_TE[0,0] = np.interp(l[i,j],ell,Cl_TT)
            Cov_TE[1,1] = np.interp(l[i,j],ell,Cl_EE)
            Cov_TE[0,1] = np.interp(l[i,j],ell,Cl_TE)
            Cov_TE[1,0] = Cov_TE[0,1]
            
            #Calculate Cholesky sqrt (since there are non-zero values for off diagonal cannot take sqrt)
            Sqrt_Cov_TE = np.linalg.cholesky(Cov_TE)
            
            #Mix X,Y arrays to create partial correlation
            mixed = np.array([random_array_X[i,j], random_array_Y[i,j]])
            
            #take inner product
            TE_FS = np.inner(Sqrt_Cov_TE, np.transpose(mixed))
            T_FS[i,j] = TE_FS[0,0]
            E_FS[i,j] = TE_FS[0,1]
            
            #output 512x512xnrlz T_FS and E_FS matrices filled in with elements calculated within loop
    
    #interpolate B and multiply by the randomized array
    B_FS = np.interp(l, ell, np.sqrt(Cl_BB), right=0)*random_array_B
    
    #Rotate E/B to Q/U (FS = fourier space), this is just rotation matrix multiplied by column matrix [[EFS], [BFS]]
    #here 2*phi is used instead of just phi -- line segment is back to itself after 180 degree rotation
    Q_FS = E_FS*np.cos(2*phi) - B_FS*np.sin(2*phi)
    U_FS = E_FS*np.sin(2*phi) + B_FS*np.cos(2*phi)
    
    #Convert T, Q, U to real space using inverse fourier transform
    #ifftshift will shift zero-frequency component to center of spectrum
    T_RS = np.fft.ifft2(np.fft.ifftshift(T_FS))
    T_map = np.real(T_RS)*npix*np.pi
    
    Q_RS = np.fft.ifft2(np.fft.ifftshift(Q_FS))
    Q_map = np.real(Q_RS)*npix*np.pi
    
    U_RS = np.fft.ifft2(np.fft.ifftshift(U_FS))
    U_map = np.real(U_RS)*npix*np.pi

    return T_map,Q_map,U_map


def power_spectra(T,Q,U, npix=512, map_size=20,bin_size=1500,bin_interval=40):
    """
    power_spectra
    
    Create power spectra from input T/Q/U maps. 
    
    Inputs
    ------
    T/Q/U: array, shape=(npix,npix).
    
    npix : integer value, optional
        Number of pixels in x and y of map. Default is 512.
    map_size: float, optional
        Map size (degrees) in x and y. Default is 20 degrees.
    bin_size: interger value, optional
        "x" range of power spectra. Default is 1500.
    bin_interval: interger value, optional
        Step size along "x", creates bin_size/bin_interval amount of "y" data. Default is 40.
            
    Outputs
    -------
    Outputs produce "y" coordinates of angular power spectra. Define a separate input for "x"
    
    DlTT, DlEE, DlBB, DlTE, DlEB, DlTB: array, shape=((bin_size/bin_interval) - 1)
    
    """

    #Transforming from real space(RS) to Fourier space(FS) using 2-d Fourier Transform,
    #A dividing factor of(np.pi*(npix**2)) is needed
    TN = np.fft.fftshift(np.fft.fft2(T))/(np.pi*(npix**2))
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
    TT=np.real(np.conj(TN)*TN)*(l*(l+1))/(2*np.pi)
    EE=np.real(np.conj(E)*E)*(l*(l+1))/(2*np.pi)
    BB=np.real(np.conj(B)*B)*(l*(l+1))/(2*np.pi)
    TE=np.real(np.conj(TN)*E)*(l*(l+1))/(2*np.pi)
    EB=np.real(np.conj(E)*B)*(l*(l+1))/(2*np.pi)
    TB=np.real(np.conj(TN)*B)*(l*(l+1))/(2*np.pi)
    
    #define bin.. goes from {0,1,2,...,bin_size} in steps of bin_interval
    bin_edge = np.arange(0, bin_size, bin_interval)
    
    #Create list of zeros filled in with mean values calculated in for loop
    DlTT = np.zeros(len(bin_edge)-1)
    DlEE = np.zeros(len(bin_edge)-1)
    DlBB = np.zeros(len(bin_edge)-1)
    DlTE = np.zeros(len(bin_edge)-1)
    DlEB = np.zeros(len(bin_edge)-1)
    DlTB = np.zeros(len(bin_edge)-1)
    
    #finding mean at values of l for all types of spectra
    for i in range(len(DlTT)):
        DlTT[i]=np.mean(TT[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlEE[i]=np.mean(EE[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlBB[i]=np.mean(BB[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlTE[i]=np.mean(TE[(l>=bin_edge[i]) & (l<bin_edge[i+1])])  
        DlEB[i]=np.mean(EB[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlTB[i]=np.mean(TB[(l>=bin_edge[i]) & (l<bin_edge[i+1])])    
        
    return DlTT, DlEE, DlBB, DlTE, DlEB, DlTB

def generate_true_QU_UM(Q,U, npix=512, map_size=20):
    """
    generate_true_QU_UM

    Create true B-mode polarization maps from input Q/U maps (in this case with partially correlated temperature/E-mode maps). 
    E-mode polarization is zeroed out in Fourier space. ***UNMASKED MAPS***
    
    Inputs
    ------
    Q : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
        
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Outputs
    -------
    Q_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    
    """
        
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #Q/U unmasked complete here. We don't need T now. 
    #We need to go to Fourier Space again. 
    #**Possible factor needed here**
    Q_FS_II = np.fft.fftshift(np.fft.fft2(Q))
    U_FS_II = np.fft.fftshift(np.fft.fft2(U))
    
    #inverted matrix to get back to E/B. We will just set E=0 
    E = (Q_FS_II*np.cos(2*phi) + U_FS_II*np.sin(2*phi))*0
    B =(-Q_FS_II*np.sin(2*phi) + U_FS_II*np.cos(2*phi))
    
    #now rotate back to Q/U 
    QT = E*np.cos(2*phi) - B*np.sin(2*phi)
    UT = E*np.sin(2*phi) + B*np.cos(2*phi)
    
    #**Possible factor needed here**
    Q_B = np.fft.ifft2(np.fft.ifftshift(QT))
    Q_true = np.real(Q_B)
    U_B = np.fft.ifft2(np.fft.ifftshift(UT))
    U_true = np.real(U_B)
    
    QU_true = np.zeros((2,npix,npix))
    QU_true[0,:,:] = Q_true
    QU_true[1,:,:] = U_true
    
    return QU_true

######## CAN DELETE LATER #######
def generate_filtered_QU(Q,U, Efilter=1,Bfilter=1,npix=512, map_size=20,sig=200):
    """
    generate_true_QU ##renamed to generate_filtered_QU

    Create true B-mode polarization maps from input Q/U maps (in this case with partially correlated temperature/E-mode maps). 
    E-mode polarization is zeroed out in Fourier space.
    
    Inputs
    ------
    Q : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
        
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Outputs
    -------
    Q_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    
    """
        
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #Q/U unmasked complete here. We don't need T now. 
    #We need to go to Fourier Space again. 
    #**Possible factor needed here**
    Q_FS_II = np.fft.fftshift(np.fft.fft2(Q))
    U_FS_II = np.fft.fftshift(np.fft.fft2(U))
    
    #inverted matrix to get back to E/B. We will just set E=0 
    E = (Q_FS_II*np.cos(2*phi) + U_FS_II*np.sin(2*phi))*Efilter
    B =(-Q_FS_II*np.sin(2*phi) + U_FS_II*np.cos(2*phi))*Bfilter #np.exp(-0.5*l**2/sig**2)
    
    #now rotate back to Q/U 
    QT = E*np.cos(2*phi) - B*np.sin(2*phi)
    UT = E*np.sin(2*phi) + B*np.cos(2*phi)
    
    #**Possible factor needed here**
    Q_B = np.fft.ifft2(np.fft.ifftshift(QT))
    Q_true = np.real(Q_B)*mask
    U_B = np.fft.ifft2(np.fft.ifftshift(UT))
    U_true = np.real(U_B)*mask
    
    QU_true = np.zeros((2,npix,npix))
    QU_true[0,:,:] = Q_true
    QU_true[1,:,:] = U_true
    
    return QU_true

def generate_filtered_QU(Q,U, Efilter=1,Bfilter=1,npix=512, map_size=20):
    """
    generate_true_QU ##renamed to generate_filtered_QU

    Create true B-mode polarization maps from input Q/U maps (in this case with partially correlated temperature/E-mode maps). 
    E-mode polarization is zeroed out in Fourier space.
    
    Inputs
    ------
    Q : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
        
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Outputs
    -------
    Q_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    
    """
        
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #Q/U unmasked complete here. We don't need T now. 
    #We need to go to Fourier Space again. 
    #**Possible factor needed here**
    Q_FS_II = np.fft.fftshift(np.fft.fft2(Q))
    U_FS_II = np.fft.fftshift(np.fft.fft2(U))
    
    #inverted matrix to get back to E/B. We will just set E=0 
    E = (Q_FS_II*np.cos(2*phi) + U_FS_II*np.sin(2*phi))*Efilter
    B =(-Q_FS_II*np.sin(2*phi) + U_FS_II*np.cos(2*phi))*Bfilter #np.exp(-0.5*l**2/sig**2)
    
    #now rotate back to Q/U 
    QT = E*np.cos(2*phi) - B*np.sin(2*phi)
    UT = E*np.sin(2*phi) + B*np.cos(2*phi)
    
    #**Possible factor needed here**
    Q_B = np.fft.ifft2(np.fft.ifftshift(QT))
    Q_true = np.real(Q_B)
    U_B = np.fft.ifft2(np.fft.ifftshift(UT))
    U_true = np.real(U_B)
    
    QU_true = np.zeros((2,npix,npix))
    QU_true[0,:,:] = Q_true
    QU_true[1,:,:] = U_true
    
    return QU_true

def generate_true_QU_sqrtmask(Q,U,npix=512,map_size=20):
    """
    generate_true_QU_sqrtmask

    Create true B-mode polarization maps from input Q/U maps (in this case with partially correlated temperature/E-mode 
    maps). E-mode polarization is zeroed out in Fourier space. Uses sqrt(mask) instead of mask
    
    Inputs
    ------
    Q : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
        
    Npix : int, optional
        Number of pixels in x and y. Default is 512.
    map_size : float, optional
        Map size (degrees) in x and y. Default is 20 degrees.

    Outputs
    -------
    Q_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    U_true : array, shape=(npix,npix)
        Polarization map generated from angular power spectrum
    
    """
        
    #Define Fourier plane coordinates.
    dl = 2.0 * np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix), np.arange(-npix/2, npix/2) * dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2 + ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0

    #Q/U unmasked complete here. We don't need T now. 
    #We need to go to Fourier Space again. 
    #**Possible factor needed here**
    Q_FS_II = np.fft.fftshift(np.fft.fft2(Q))
    U_FS_II = np.fft.fftshift(np.fft.fft2(U))
    
    #inverted matrix to get back to E/B. We will just set E=0 
    E = (Q_FS_II*np.cos(2*phi) + U_FS_II*np.sin(2*phi))*0
    B =(-Q_FS_II*np.sin(2*phi) + U_FS_II*np.cos(2*phi))
    
    #now rotate back to Q/U 
    QT = E*np.cos(2*phi) - B*np.sin(2*phi)
    UT = E*np.sin(2*phi) + B*np.cos(2*phi)
    
    #**Possible factor needed here**
    Q_B = np.fft.ifft2(np.fft.ifftshift(QT))
    Q_true = np.real(Q_B)*np.sqrt(mask)
    U_B = np.fft.ifft2(np.fft.ifftshift(UT))
    U_true = np.real(U_B)*np.sqrt(mask)
    
    QU_true = np.zeros((2,npix,npix))
    QU_true[0,:,:] = Q_true
    QU_true[1,:,:] = U_true
    
    return QU_true

def generate_maps(ell,Dl_TT,Dl_EE,Dl_BB,Dl_TE,nrlz,npix=512,map_size=20):
    CMB_maps=np.zeros((3,npix,npix,nrlz))
    for i in range(nrlz):
        T,Q,U=generate_CMB_maps(ell,Dl_TT,Dl_EE,Dl_BB,Dl_TE)
        CMB_maps[0,:,:,i]=T
        CMB_maps[1,:,:,i]=Q
        CMB_maps[2,:,:,i]=U
    
    return CMB_maps

    
def cross_spectra(Q1,U1,Q2,U2, npix=512, map_size=20,bin_size=1500,bin_interval=40):
    """
    power_spectra
    
    Create power spectra from input T/Q/U maps. 
    
    Inputs
    ------
    T/Q/U: array, shape=(npix,npix).
    
    npix : integer value, optional
        Number of pixels in x and y of map. Default is 512.
    map_size: float, optional
        Map size (degrees) in x and y. Default is 20 degrees.
    bin_size: interger value, optional
        "x" range of power spectra. Default is 1500.
    bin_interval: interger value, optional
        Step size along "x", creates bin_size/bin_interval amount of "y" data. Default is 40.
            
    Outputs
    -------
    Outputs produce "y" coordinates of angular power spectra. Define a separate input for "x"
    
    DlTT, DlEE, DlBB, DlTE, DlEB, DlTB: array, shape=((bin_size/bin_interval) - 1)
    
    """

    #Transforming from real space(RS) to Fourier space(FS) using 2-d Fourier Transform,
    #A dividing factor of(np.pi*(npix**2)) is needed
    QN1 = np.fft.fftshift(np.fft.fft2(Q1))/(np.pi*(npix**2))
    UN1 = np.fft.fftshift(np.fft.fft2(U1))/(np.pi*(npix**2))
    
    QN2 = np.fft.fftshift(np.fft.fft2(Q2))/(np.pi*(npix**2))
    UN2 = np.fft.fftshift(np.fft.fft2(U2))/(np.pi*(npix**2))
    
    
    #define Fourier plane coordinates
    dl = 2.0*np.pi / np.radians(map_size)
    lx = np.outer(np.ones(npix),np.arange(-npix/2,npix/2)*dl)
    ly = np.transpose(lx)
    l = np.sqrt(lx**2+ly**2)
    phi = np.arctan2(ly,lx)
    phi[l == 0] = 0.0
    
    #Inverted rotation matrix 
    #line segments of E and B mode are symmetric over pi radians
    E1 = QN1*np.cos(2*phi) + UN1*np.sin(2*phi)
    B1 =-QN1*np.sin(2*phi) + UN1*np.cos(2*phi)
    
    E2 = QN2*np.cos(2*phi) + UN2*np.sin(2*phi)
    B2 =-QN2*np.sin(2*phi) + UN2*np.cos(2*phi)
    
    
    #apply conjugate and doing operation to convert Cl to Dl   
    BB11=np.real(np.conj(B1)*B1)*(l*(l+1))/(2*np.pi)
    BB12=np.real(np.conj(B1)*B2)*(l*(l+1))/(2*np.pi)
    BB21=np.real(np.conj(B2)*B1)*(l*(l+1))/(2*np.pi)
    BB22=np.real(np.conj(B2)*B2)*(l*(l+1))/(2*np.pi)

    #define bin.. goes from {0,1,2,...,bin_size} in steps of bin_interval
    bin_edge = np.arange(0, bin_size, bin_interval)
    
    #Create list of zeros filled in with mean values calculated in for loop
    DlBB11 = np.zeros(len(bin_edge)-1)
    DlBB12 = np.zeros(len(bin_edge)-1)
    DlBB21 = np.zeros(len(bin_edge)-1)
    DlBB22 = np.zeros(len(bin_edge)-1)

    #finding mean at values of l for all types of spectra
    for i in range(len(DlBB11)):
        DlBB11[i]=np.mean(BB11[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlBB12[i]=np.mean(BB12[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlBB21[i]=np.mean(BB21[(l>=bin_edge[i]) & (l<bin_edge[i+1])])
        DlBB22[i]=np.mean(BB22[(l>=bin_edge[i]) & (l<bin_edge[i+1])])  
        
    return DlBB11, DlBB12, DlBB21, DlBB22

def correlation(prediction,truth_maps,mask):
    """
    Input arrays of prediction maps created by CNN model and truth maps to compare prediction maps to.
    Averages over values of BB11, BB12, BB22. (1 represents input, prediction map from power spectra, 2 represents truth map from power 
    spectra)
    Calculates correlation between BB for 1 and 2. 
    
    Inputs
    ------
    prediction: array, shape = (#models, #validation maps, npix, npix, 2), {2 from polarization type either Q/U}
    truth_maps: array, shape = (#models, #validation maps, npix, npix, 2)
    
    Outputs
    ------
    correlation: array, shape = (#models, correlation values)
    """
    #selecting size of number of models & maps
    models = prediction.shape[0]
    maps = prediction.shape[1]
    
    #set zeros array 
    DlBB11_pred=np.zeros((models,maps,37))
    DlBB12_pred=np.zeros((models,maps,37))
    DlBB21_pred=np.zeros((models,maps,37))
    DlBB22_pred=np.zeros((models,maps,37))
    
    #input here is Q/U maps from the prediction/truth_maps. 
    #cross_spectra will calculate the cross spectrum of these in the form of DlBB11, DlBB12, DlBB21, DlBB2.
    
    for i in range(maps):
        for j in range(models):
            DlBB11_pred[j,i,:], DlBB12_pred[j,i,:], DlBB21_pred[j,i,:],DlBB22_pred[j,i,:]=cross_spectra(
                prediction[j,i,:,:,0],prediction[j,i,:,:,1],truth_maps[i,:,:,0],truth_maps[i,:,:,1])
    
    mean11=np.zeros((models,37))
    mean12=np.zeros((models,37))
    mean22=np.zeros((models,37))
    
    for i in range(models):
        mean11[i,:]=np.mean(DlBB11_pred[i,:,:]/np.mean(mask**2),axis=0)
    for i in range (models):
        mean12[i,:]=np.mean(DlBB12_pred[i,:,:]/np.mean(mask**2),axis=0)
    for i in range(models):
        mean22[i,:]=np.mean(DlBB22_pred[i,:,:]/np.mean(mask**2),axis=0)
    
    correlation = mean12/(np.sqrt(mean11)*np.sqrt(mean22)) #normalization factors not needed as they divide out with this 
                                                           #calculation
    
    return correlation

