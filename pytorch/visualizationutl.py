

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageutl as imutl


def display_random_images(images):
    """
    Display random images from dataset
    For debug only
    """

    fig, ax = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

    for i in range(3):
        for j in range(3):
            rand_idx = np.random.randint(images.num)
            image = images.getimage(rand_idx)
            ax[i,j].imshow(image)
            ax[i,j].set_title('Image Idx: %d' % (rand_idx,))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def display_image_and_gt( images ):
    '''
    Display random image and gt
    '''
    rand_idx = np.random.randint(images.num)
    
    image = images.getimage(rand_idx)
    label = images.getlabel()
    mask = cv2.bitwise_and(image, image, mask=label)
    
    fig, ax = plt.subplots(1, 4, figsize=(10, 10), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
    
    ax[0].imshow(image); ax[0].set_title('Image')
    ax[1].imshow(label); ax[1].set_title('Label')
    ax[2].imshow(image); 
    ax[2].imshow(label, 'jet', interpolation='none', alpha=0.7); ax[2].set_title('Overlap')
    ax[3].imshow(mask); ax[3].set_title('Mask');
    
    for a in ax.ravel():
        a.set_axis_off()
    
    plt.tight_layout()
    plt.show()




