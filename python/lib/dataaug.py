'''
Augment the data
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageutl as imutl
import numpy as np
import cv2
import os

import skimage.color as skcolor
import skimage.util as skutl
import scipy.misc

import PIL.Image as Image

# Image data augmentation parameters
ANGLE = 15
TRANSLATION = 0.2
WARP = 0.0  #0.05
#NUM_NEW_IMAGES = 100000
NUM_NEW_IMAGES = 1000
SIZE = 512

########################################################
# Helper functions
########################################################

def transform_size(image, label, size=512):

    height, width, channels = image.shape;

    image = np.array(image)
    label = np.array(label);

    asp = float(height)/width;
    w = size; 
    h = int(w*asp);

    image_x = scipy.misc.imresize(image.copy(), (h,w), interp='bilinear');
    label_x = scipy.misc.imresize(label.copy(), (h,w), interp='nearest', mode='F');

    image = np.zeros((w,w,3));
    label = np.zeros((w,w));

    ini = int(round((w-h) / 2.0));
    image[ini:h+ini,:,:] = image_x;
    label[ini:ini+h,:] = label_x;

    image = image.astype(np.uint8)
    label = label.astype(np.uint8)
    label[label>0] = 255;

    _,label = cv2.threshold(label,127,255,0);
    _,contours,_ = cv2.findContours(label, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
    for cnt in contours: cv2.drawContours(label, cnt, -1, 2, 1)

    return image, label;


def transform_image(image, label, angle, translation, warp):
    """
    Transform the image for data augmentation

    Arguments:
        * image: Input image
        * angle: Max rotation angle, in degrees. Direction of rotation is random.
        * translation: Max translation amount in both x and y directions,
            expressed as fraction of total image width/height
        * warp: Max warp amount for each of the 3 reference points,
            expressed as fraction of total image width/height

    Returns:
        * Transformed image as an np.array() object
    """

    height, width, channels = image.shape
    
    image = np.array(image)
    label = np.array(label)


    # Rotation
    #center = (width//2, height//2)
    #angle_rand = np.random.uniform(-angle, angle)
    #rotation_mat = cv2.getRotationMatrix2D(center, angle_rand, 1)
    #image = cv2.warpAffine(image, rotation_mat, (width, height))

    # Translation
    x_offset = translation * width * np.random.uniform(-1, 1)
    y_offset = translation * height * np.random.uniform(-1, 1)
    translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])

    image = cv2.warpAffine(image, translation_mat, (width, height))
    label = cv2.warpAffine(label, translation_mat, (width, height))


    # Horizontal mirror
    if bool(np.random.randint(0,2)):
        image = np.fliplr(image)
        label = np.fliplr(label)
	

    # Warp
    # NOTE: The commented code below is left for reference
    # The warp function tends to blur the image, so it is not useds
    '''
    src_triangle = np.float32([[0, 0], [0, height], [width, 0]])
    x_offsets = [warp * width * np.random.uniform(-1, 1) for _ in range(3)]
    y_offsets = [warp * height * np.random.uniform(-1, 1) for _ in range(3)]
    dst_triangle = np.float32([[x_offsets[0], y_offsets[0]],\
                             [x_offsets[1], height + y_offsets[1]],\
                             [width + x_offsets[2], y_offsets[2]]])
    warp_mat = cv2.getAffineTransform(src_triangle, dst_triangle)
    
    image = cv2.warpAffine(image, warp_mat, (width, height))
    '''

    # Global illumination change
    image_lab = skcolor.rgb2lab(image)
    image_lab[:,:,0] = image_lab[:,:,0]*(np.random.rand(1)+0.5)
    image_lab[:,:,0] = np.clip(image_lab[:,:,0],0,100)
    image = skcolor.lab2rgb(image_lab)

    # Gaussian noise
    if bool(np.random.randint(0,2)):
        image = skutl.random_noise(image, mode='gaussian', seed=None, clip=True)
    
    # Gaussian blur
    if bool(np.random.randint(0,2)):
        wnd = np.random.randint(1, 3) * 2 + 1
        image = cv2.GaussianBlur(image, (wnd,wnd), 0);

    #image = (image).astype(np.uint8)
    label = (label==255).astype(np.uint8)

    return image, label


def display_random_images_aug(data):
    """
    Display random image, and transformed versions of it
    For debug only
    """

    image = data.getimage(np.random.randint(data.num));
    label = data.getlabel();    

    fig, ax = plt.subplots(3, 4, figsize=(10, 6), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})

    for i in range(3):
        
        image_x = image; 
        label_x = label;
        image_x, label_x = transform_size(image_x, label_x, SIZE);        
        image_x, label_x = transform_image(image_x, label_x, ANGLE, TRANSLATION, WARP)

        ax[i,0].imshow(image_x); ax[i,0].set_title('Image')
        ax[i,1].imshow(image_x); ax[i,1].set_title('Transformed image')
        ax[i,2].imshow(label_x); ax[i,2].set_title('Transformed label')
        ax[i,3].imshow(image_x); 
        ax[i,3].imshow(label_x, 'jet', interpolation='none', alpha=0.7); ax[i,3].set_title('Transformed overlap')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()



def getpalette():
    mpalette = [255 for x in range(769)]
    mpalette[0:3] = [0,0,0];
    mpalette[3:6] = [255,0,0];
    mpalette[6:8] = [0,255,0];
    return mpalette


########################################################
# Main function
########################################################
def data_aug(orig_file, new_file, num_new_images=1000, size=512):
    """
    blah
    """

    # Load original dataset
    data = imutl.dataProvide( orig_file, fn_image='train', fn_label='train_masks', lext='gif');
    if data.isempty(): return;

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(num_new_images):

        # Pick a random image from original dataset to transform
        rand_idx = np.random.randint(data.num);
        image = data.getimage(rand_idx);
        label = data.getlabel();

        # Create new image       
        #image_x, label_x = transform_size(image, label, size);
        image_x, label_x = transform_image(image, label, ANGLE, TRANSLATION, WARP)
        image_x, label_x = transform_size(image_x, label_x, size)  

        label_x[label_x == 255] = 1;
        label_x[label_x == 2] = 255;   
        label_x = Image.fromarray(np.uint8(label_x), mode='P');             

        # Add new data to augmented dataset
        scipy.misc.imsave(os.path.join(new_file, 'images', 'aug{:06d}.png'.format(i)), image_x);
        #scipy.misc.imsave(os.path.join(new_file, 'labels', 'aug{:06d}.png'.format(i)), label_x);
        
        label_x.putpalette(getpalette())
        label_x.save(os.path.join(new_file, 'labels', 'aug{:06d}.png'.format(i)))


        if (i+1) % 10 == 0:
            print('%d new images generated' % (i+1,))


def data_export(orig_file, new_file, size=512):
    """
    blah
    """

    # Load original dataset
    data = imutl.dataProvide( orig_file, fn_image='train', fn_label='train_masks', lext='gif');
    if data.isempty(): return;

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(data.num):

        # Pick a random image from original dataset to transform
        rand_idx = i;
        image = data.getimage(rand_idx);
        label = data.getlabel();

        # Create new image       
        #image_x, label_x = transform_size(image, label, size);
        #image_x, label_x = transform_image(image, label, ANGLE, TRANSLATION, WARP)
        image_x, label_x = transform_size(image, label, size)  

        label_x[label_x == 255] = 1;
        label_x[label_x == 2] = 255;   
        label_x = Image.fromarray(np.uint8(label_x), mode='P');             

        # Add new data to augmented dataset
        scipy.misc.imsave(os.path.join(new_file, 'images', 'aug{:06d}.png'.format(i)), image_x);
        #scipy.misc.imsave(os.path.join(new_file, 'labels', 'aug{:06d}.png'.format(i)), label_x);
        
        label_x.putpalette(getpalette())
        label_x.save(os.path.join(new_file, 'labels', 'aug{:06d}.png'.format(i)))


        if (i+1) % 10 == 0:
            print('%d new images generated' % (i+1,))





if __name__ == '__main__':

    path = '../db';
    namedataset = 'car';
    namedataset_out = 'car_aug';
    num_new_images = NUM_NEW_IMAGES;

    root = os.path.join(path,namedataset);
    root_out = os.path.join(path,namedataset_out);

    if os.path.exists(root_out) is not True:
        os.makedirs(root_out);
        os.makedirs(os.path.join(root_out,'images'));
        os.makedirs(os.path.join(root_out,'labels'));

    # This actually creates the augmented dataset
    #data_aug(root, root_out, num_new_images)
    data_export(root, root_out, 1024);

