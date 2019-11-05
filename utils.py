import matplotlib.pyplot as plt
import csv, os
import numpy as np
import cv2

def normalize_input(ux, uy, image_size):
    width = image_size[1]
    height = image_size[0]

    cx = width / 2.0
    cy = height / 2.0

    nx = (ux - cx) / width
    ny = (uy - cy) / height

    return nx, ny


def unnormalize_input(ux, uy, image_size):
    width = image_size[1]
    height = image_size[0]

    cx = width / 2.0
    cy = height / 2.0

    unx = int(width * ux + cx)
    uny = int(height * uy + cy)

    return unx, uny

def normalize_input_vec(u, image_size):
    uix = u[0]
    uiy = u[1]
    ufx = u[2]
    ufy = u[3]
    new_uix, new_uiy = normalize_input(uix, uiy, image_size)
    new_ufx, new_ufy = normalize_input(ufx, ufy, image_size)
    return np.array([new_uix, new_uiy, new_ufx, new_ufy])

def unnormalize_input_vec(u, image_size):
    uix = u[0]
    uiy = u[1]
    ufx = u[2]
    ufy = u[3]
    new_uix, new_uiy = unnormalize_input(uix, uiy, image_size)
    new_ufx, new_ufy = unnormalize_input(ufx, ufy, image_size)
    return np.array([new_uix, new_uiy, new_ufx, new_ufy])

# Input are unnormalized numpy arrays
def construct_bound(ui, uf, pusher_width):
    height = np.linalg.norm(uf - ui)
    # Construct coordinates before rotation
    coords = np.array([[-pusher_width/2, height], [-pusher_width/2, 0],\
             [pusher_width/2, 0], [pusher_width/2, height]])
    new_coords = np.zeros((4,2))

    theta = np.arctan2((uf-ui)[1], (uf-ui)[0]) - np.pi / 2 
    R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]

    for i in range(len(coords)):
        coord = np.transpose(coords[i])
        new_coords[i,:] = np.dot(R, coord) + np.transpose(np.array(ui))

    return new_coords

# Make sure images are black and white
def validation_image(image_i, image_f, u, mode):
    width, height = image_i.shape

    uix = u[0]
    uiy = u[1]
    ufx = u[2]
    ufy = u[3]
    
    uix, uiy = unnormalize_input(uix, uiy, image_i.shape)
    ufx, ufy = unnormalize_input(ufx, ufy, image_f.shape)

    coords = construct_bound(np.array([uix, uiy]), np.array([ufx, ufy]), 22.5)
    image_rgb = np.zeros((width, height, 3), dtype=np.uint8)

    thres = 100

    for i in range(width):
        for j in range(height):
            if (image_i[i,j] > thres) and (image_f[i,j] > thres):
                image_rgb[i,j,:] = [0, 255, 128]
            elif (image_i[i,j] > thres): 
                image_rgb[i,j,:] = [0, 128, 255]
            elif (image_f[i,j] > thres):
                image_rgb[i,j,:] = [255, 0, 128]
            else:
                image_rgb[i,j,:] = [255, 255, 255]
                
    plt.figure()
    plt.imshow(image_rgb, alpha=1.0)
    plt.plot(uix, uiy, 'bo')
    plt.plot(ufx, ufy, 'ro')
    plt.plot([uix, ufx], [uiy, ufy], 'k--')
    plt.fill(coords[:,0], coords[:,1], color="green", alpha=0.1)
    #plt.tick_params(axis='both', which='both', bottom=False, top=False,\
    #                left=False, right=False, labelbottom=False, labelleft=False)

    plt.axis([0, height, 0, width])

    if (mode[0] == "display"):
        plt.show()
    elif (mode[0] == "save"):
        plt.savefig(mode[1])
        plt.close()
    else:
        raise ValueError('Wrong Value. Use display or save mode')


    

            



    

