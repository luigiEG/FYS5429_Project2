import matplotlib.pyplot as plt
import numpy as np
import torch
import Pk_library as PKL





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









# wrapper to work with dataloader/dataset

# define the funcrion mean_PS_from_dataloader(data_loader, model=None)
# if model is None, then obtain the mean PS from the dataloader
# if model is not None, then obtain the mean PS from the reconstructed images

def mean_PS_from_dataloader(dataloader, model=None):
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
    return np.mean(ps, axis=0)

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






