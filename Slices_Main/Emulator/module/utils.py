



import matplotlib.pyplot as plt
import numpy as np
import torch
import Pk_library as PKL

## Constants:
a = 8  # constant value for calculations

min_val = -1.3  # minimum value for normalization and denormalization
max_val = 0.97  # maximum value for normalization and denormalization

#######################################################################
## Data Normalization and Denormalization functions:
#######################################################################
def denorm(y):
    """
    Denormalizes the data using the provided min and max values.

    Parameters:
    y (float): Normalized data to be denormalized.

    Returns:
    float: Denormalized data.
    """
    return (y * (max_val - min_val) + min_val)

def anti_s2(y):
    """
    Performs the anti-s2 transformation on the input data.

    Parameters:
    y (float): Input data to be transformed.

    Returns:
    float: Transformed data.
    """
    return a * (y + 1) / (1 - y)

def normalization(x):
    """
    Normalizes the data within the range of [-1, 1].

    Parameters:
    x (float): Data to be normalized.

    Returns:
    float: Normalized data.
    """
    return (x - min_val) / (max_val - min_val)

def s2(x):
    """
    Performs the s2 transformation on the input data.

    Parameters:
    x (float): Input data to be transformed.

    Returns:
    float: Transformed data.
    """
    return (2 * x / (x + a)) - 1






#######################################################################
## Data import 
#######################################################################

def txt_loader(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.strip().split())) for line in lines]
        array = np.array(data,dtype='float32')
    return normalization(s2(array))


def npy_loader(path):
    matrix = np.load(path)
    return normalization(s2(matrix.astype('float32')))















#######################################################################
## Powerspectrum functions:
#######################################################################

def PowerSpectrum(normalized_data):
    delta = denorm(anti_s2(normalized_data))
    # parameters
    grid    = 128     #the map will have grid^2 pixels
    BoxSize = 512.0  #Mpc/h
    MAS     = 'None'#'CIC'  #MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
    threads = 1       #number of openmp threads
    # compute the Pk of the image
    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads)
    # get the attributes of the routine
    k      = Pk2D.k      #k in h/Mpc
    Pk     = Pk2D.Pk     #Pk in (Mpc/h)^2
    Nmodes = Pk2D.Nmodes #Number of modes in the different k bins
    return np.column_stack((k,Pk))

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
    PS_recon = np.array([PowerSpectrum(image) for image in recon_images])
    PS_train = np.array([PowerSpectrum(image) for image in train_images])
    mean_train_PS = calculate_mean_PS(PS_train)
    mean_recon_PS = calculate_mean_PS(PS_recon)

    return np.sqrt(np.mean(np.square(mean_recon_PS[:,1] - mean_train_PS[:,1])))








#######################################################################
# wrapper to work with dataloader/dataset
#######################################################################
# define the funcrion mean_PS_from_dataloader(data_loader, model=None)
# if model is None, then obtain the mean PS from the dataloader
# if model is not None, then obtain the mean PS from the reconstructed images
# define the function mean_PS_from_dataloader
def mean_PS_from_dataloader(dataloader, model=None):
    if model is None:  # if model is not provided
        ps = []
        for i, (images, _) in enumerate(dataloader):
            ps.append(PowerSpectrum(images[0][0].numpy()))  # calculate Power Spectrum from original images
    else:  # if model is provided
        ps = []
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                recon_images, _, _ = model(images)  # reconstruct images using the model
                recon_image = recon_images[0][0].numpy()  # obtain reconstructed image in numpy format
                ps.append(PowerSpectrum(recon_image))  # calculate Power Spectrum from reconstructed images
    return np.mean(ps, axis=0)  # return the mean Power Spectrum

def std_PS_from_dataloader(dataloader, model=None):
    if model is None:
        ps = []
        for i, (images, _) in enumerate(dataloader):
            ps.append(PowerSpectrum(images[0][0].numpy()))
    else:
        ps = []
        with torch.no_grad():

            for i, (images, _) in enumerate(dataloader):
                recon_images, _, _ = model(images)
                recon_image = recon_images[0][0].numpy()
                ps.append(PowerSpectrum(recon_image))
    return np.mean(np.std(ps, axis=0))





# get the mse between each pair of images
def get_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# get the mse between each pair of images in the dataset
def get_mse_matrix(dataset):
    n = len(dataset)
    mse_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # chech if is a tensor or numpu array
            if isinstance(dataset[i][0], torch.Tensor):
                mse_matrix[i, j] = get_mse(dataset[i][0].numpy(), dataset[j][0].numpy())
            else:
                mse_matrix[i, j] = get_mse(dataset[i][0], dataset[j][0])
    return mse_matrix

# get the mean mse between each pair of images in the dataset
def get_mean_mse(dataset):
    # only upper triangular part of the matrix
    mse_matrix = get_mse_matrix(dataset)
    mse = np.mean(mse_matrix[np.triu_indices(len(dataset), k=1)])
    return mse

# get mse flat from dataset, using only the upper triangular part of the matrix
def get_mse_flat(dataset):
    mse_matrix = get_mse_matrix(dataset)
    return mse_matrix[np.triu_indices(len(dataset), k=1)]
