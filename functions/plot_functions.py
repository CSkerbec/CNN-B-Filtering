import numpy as np 
import matplotlib.pyplot as plt

#plots T/Q/U maps
def plot_map(Map,color_range=(-5,5),Title="",xlabel="",ylabel="",cbarlabel='$\mu$K'):
    fig = plt.figure()
    imgplot = plt.imshow(Map,extent=(0,20,0,20),vmin=color_range[0],vmax=color_range[1])
    plt.title(Title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel(cbarlabel,rotation=270)

#plots power maps
def plot_power(Dl,ell=0,Dl_true=0,Title=""):
    """
    plot_power(Dl,ell=0,Dl_true=0,Title="")
    
    Input: array, shape(37)
    """
    #defining x coordinate
    x=np.arange(0,1460,40)+20

    #plotting power spectra 
    plt.plot(x,Dl)
    plt.plot(ell,Dl_true)
    plt.xlabel('Multipole moment, $\ell$');
    plt.ylabel('$\ell (\ell+1) C_\ell / 2 \pi(\mu K^2)$')
    plt.title(Title)

#plots correlation maps 
def plot_corr(correlated,model_maximum=100,Title="", polarization="B"):
    plt.figure()
    multiple=int(model_maximum/correlated.shape[0])
    labels = []
    for i in range (1,multiple+1): 
        labels.append("e" + str(i*multiple))
    for i in range(correlated.shape[0]):
        plt.plot(np.arange(0,1460,40)+20,correlated[i,:],label=labels[i])
    plt.title(polarization + " Correlation: " + Title)
    plt.xlabel('Multipole moment, $\ell$')
    plt.legend()
    plt.show()
    