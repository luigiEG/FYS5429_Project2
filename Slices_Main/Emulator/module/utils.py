import matplotlib.pyplot as plt
import numpy as np
import torch





## Constants:

a = 8

min = -1.3
max = 0.97













## Data Normalization and Denormalization functions:

def denorm(y):
    return (y*(max-min)+min)
def anti_s2(y):
    return a * (y+1)/(1-y)

def normalization(x) :
    return (x-min)/(max-min)
def s2(x):
    return (2*x / (x + a)) - 1

def binavg(pk,k_min, k_max, kgrid,nkbins):
    '''
    Bin averaging for the powerspectrum calculations
    '''
    kgrid[0,0] = 1.0
    ikbin = np.digitize(kgrid,np.linspace(k_min,k_max,nkbins+1),right=False)

    nmodes,pkavg,kmean = np.zeros(nkbins,dtype=int),np.full(nkbins,0.),np.full(nkbins,0.)
    for ik in range(nkbins):
        nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (nmodes[ik] > 0):
            pkavg[ik] = np.mean(pk[ikbin == ik+1])
            kmean[ik] = np.mean(kgrid[ikbin == ik+1])

    return pkavg, nmodes, kmean












## Data import 

def txt_loader(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
        array = np.array(data,dtype='float32')
    return normalization(array)


def npy_loader(path):
    matrix = np.load(path)
    return normalization(s2(matrix.astype('float32')))
















## Powerspectrum functions:

def binavg(pk,k_min, k_max, kgrid,nkbins):
    '''
    Bin averaging for the powerspectrum calculations
    '''
    kgrid[0,0] = 1.0
    ikbin = np.digitize(kgrid,np.linspace(k_min,k_max,nkbins+1),right=False)

    nmodes,pkavg,kmean = np.zeros(nkbins,dtype=int),np.full(nkbins,0.),np.full(nkbins,0.)
    for ik in range(nkbins):
        nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
        if (nmodes[ik] > 0):
            pkavg[ik] = np.mean(pk[ikbin == ik+1])
            kmean[ik] = np.mean(kgrid[ikbin == ik+1])

    return pkavg, nmodes, kmean

def PS(density_field):
    data = (anti_s2(denorm(density_field)))
    ## Defining the grid in 2-D:
    kx = 2. * np.pi * np.fft.fftfreq(128, d=1.)
    ky = 2. * np.pi * np.fft.fftfreq(128, d=1.)
    kgrid = np.sqrt(kx[:,np.newaxis]**2.0 + kx[np.newaxis,:]**2.0)

    k_min = 0.0
    k_max = 3.14   ## k_max = 128*pi/512 ##the Nyquist frequency calculated using the number of pixes * pi/physical length of the array
    nkbins = 16    ## to find and estimate for this calculate the fundamental frequency kf = 2*pi/512^1/3
                  ## --> in our case kf = 2*pi/Area_box^{1/2}. And then kmax-kmin/kf ~ nkbins

  ## For data1:
    delta_r = data
    delta_k = np.fft.fftn(delta_r)
    pk = np.real(delta_k * np.conj(delta_k))
    pkavg, nmodes, kmean = binavg(pk, k_min, k_max, kgrid, nkbins)
    return np.column_stack((kmean, pkavg))

def calculate_mean_PS(PS_matrix):
    pk_matrix = PS_matrix[:, :, 1]  # Extract the second value of the third dimension
    pk_mean = np.mean(pk_matrix, axis=0)
    k = PS_matrix[0, :, 0]  # Extract the first column of the first row
    mean_PS = np.column_stack((k, pk_mean))
    return mean_PS


def PS_loss(model_VAE,test_images,train_images):
    recon_images, _, _ = model_VAE(test_images) 
    recon_images = recon_images.detach().numpy().squeeze() 
    train_images = train_images.detach().numpy().squeeze() 
    PS_recon = np.array([PS(image) for image in recon_images])
    PS_train = np.array([PS(image) for image in train_images])
    mean_train_PS = calculate_mean_PS(PS_train)
    mean_recon_PS = calculate_mean_PS(PS_recon)

    return np.sqrt(np.mean(np.square(mean_recon_PS[:,1] - mean_train_PS[:,1])))



def powerspectrum_i(data, color, alpha = 0.8, label = None):
    data = (anti_s2(denorm(data)))
    ## Defining the grid in 2-D:
    kx = 2. * np.pi * np.fft.fftfreq(128, d=1.)
    ky = 2. * np.pi * np.fft.fftfreq(128, d=1.)
    kgrid = np.sqrt(kx[:,np.newaxis]**2.0 + kx[np.newaxis,:]**2.0)

    k_min = 0.0
    k_max = 3.14   ## k_max = 128*pi/512 ##the Nyquist frequency calculated using the number of pixes * pi/physical length of the array
    nkbins = 16    ## to find and estimate for this calculate the fundamental frequency kf = 2*pi/512^1/3
  	              ## --> in our case kf = 2*pi/Area_box^{1/2}. And then kmax-kmin/kf ~ nkbins

	## For data1:

    delta_r = data
    delta_k = np.fft.fftn(delta_r)
    pk = np.real(delta_k * np.conj(delta_k))
    pkavg, nmodes, kmean = binavg(pk,k_min, k_max, kgrid,nkbins)

    h = 0.7
    # Plotting the results:

    plt.figure(1,figsize=(5.5,4.5))
    plt.plot(kmean,pkavg, color = color, linewidth=1.2, alpha = alpha, label = label) ## Need to figure out the units here
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k$ [h $\mathrm{Mpc^{-1}}$]', fontsize = 13)
    plt.ylabel(r'$P(k)$', fontsize = 13)
    plt.tick_params(right=True, top=True, direction='in', which = 'both')
    plt.tick_params(which='major', length=4)
    plt.tight_layout()
    plt.legend()





# wrapper to work with dataloader/dataset

# define the funcrion mean_PS_from_dataloader(data_loader, model=None)
# if model is None, then obtain the mean PS from the dataloader
# if model is not None, then obtain the mean PS from the reconstructed images

def mean_PS_from_dataloader(dataloader, model=None):
    if model is None:
        ps = []
        for i, (images, _) in enumerate(dataloader):
            ps.append(PS(images[0][0].numpy()))
    else:
        ps = []
        with torch.no_grad():

            for i, (images, _) in enumerate(dataloader):
                recon_images, _, _ = model(images)
                recon_image = recon_images[0][0].numpy()
                ps.append(PS(recon_image))
    return np.mean(ps, axis=0)







