
import numpy as np
import scipy.misc
import skfmm
import math
import pydensecrf.densecrf as dcrf

def transform_image_size(image, size):
    '''
    Transform image size  
    '''    
    height, width, channels = image.shape;
    image = np.array(image)
    
    asp = float(height)/width;
    w = size; 
    h = int(w*asp);
    
    image_x = scipy.misc.imresize(image, (h,w), interp='bilinear');
    image = np.zeros((w,w,3));
    ini = int(round((w-h) / 2.0));
    image[ini:h+ini,:,:] = image_x;
    image = image.astype(np.uint8);

    return image, asp;

def unet_transform_image_size(image, size=388):
    
    height, width, channels = image.shape;
    image = np.array(image)
    
    asp = float(height)/width;
    w = size; 
    h = int(w*asp);
    
    image_x = scipy.misc.imresize(image.copy(), (h,w), interp='bilinear');
    image = np.zeros((w,w,3));
    ini = int(round((w-h) / 2.0));
    image[ini:ini+h,:,:] = image_x;


    image = image/255.0
    downsampleFactor = 16;
    d4a_size= 0;
    padInput =   (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
    padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;
    d4a_size = math.ceil( (size - padOutput)/downsampleFactor);
    input_size = downsampleFactor*d4a_size + padInput;
    offset=(input_size-size)//2
    image_f = np.zeros((input_size,input_size,3));
    image_f[offset:-offset,offset:-offset,:]=image;

    
    return image_f, asp, image;


def inv_transform_image_size(image, imshape, asp):
    '''
    Invert transform image size
    '''
    
    height, width = image.shape;
    image = np.array(image);    
    w = height; 
    h = int(width*(asp));    
    ini = int(round((w-h) / 2.0));
    image_x = image[ini:h+ini,:];
    image_x = scipy.misc.imresize(image_x, imshape, interp='nearest', mode='F');
    
    return image_x;

def inv_transform_label_size(image, imshape, asp):
    '''
    Invert transform image size
    '''
    
    height, width = image.shape;
    image = np.array(image);    
    w = height; 
    h = int(width*(asp));    
    ini = int(round((w-h) / 2.0));
    image_x = image[ini:h+ini,:];
    image_x = scipy.misc.imresize(image_x, imshape, interp='bilinear');
    
    return image_x;


def dicecoef(label, prediction):
    '''
    Dice coefficient
    '''
    #label = bottom[1].data[:,0,:,:]
    # compute prediction
    #prediction = np.argmax(bottom[0].data, axis=1)
    # area of predicted contour
    a_p = np.sum(prediction, axis=(0,1))
    # area of contour in label
    a_l = np.sum(label, axis=(0,1))
    # area of intersection
    a_pl = np.sum(prediction * label, axis=(0,1))
    # dice coefficient
    dice_coeff = np.mean(2.*a_pl/(a_p + a_l))
    
    return dice_coeff


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)



def process_data( class_data ):
    '''
    Precess data
    '''
    # assume the only output is a CHW image where C is the number
    # of classes, H and W are the height and width of the image
    #class_data = net_output[net_output.keys()[0]].astype('float32')
    #class_data = class_data[0,:,:,:];

    # retain only the top class for each pixel
    class_data = np.argmax(class_data, axis=0).astype('uint8')

    # remember the classes we found
    found_classes = np.unique(class_data)

    fill_data = np.ndarray((class_data.shape[0], class_data.shape[1], 4), dtype='uint8')
    for x in xrange(3):
        fill_data[:, :, x] = class_data.copy()

    # Assuming that class 0 is the background
    mask = np.greater(class_data, 0)
    fill_data[:, :, 3] = mask * 255
    line_data = fill_data.copy()
    seg_data = fill_data.copy()
    
    # Black mask of non-segmented pixels
    mask_data = np.zeros(fill_data.shape, dtype='uint8')
    mask_data[:, :, 3] = (1 - mask) * 255

    # Generate outlines around segmented classes
    if len(found_classes) > 1:
        
        # Assuming that class 0 is the background.
        line_mask = np.zeros(class_data.shape, dtype=bool)
        max_distance = np.zeros(class_data.shape, dtype=float) + 1
        for c in (x for x in found_classes if x != 0):
            c_mask = np.equal(class_data, c)
            # Find the signed distance from the zero contour
            distance = skfmm.distance(c_mask.astype('float32') - 0.5)
            # Accumulate the mask for all classes
            line_width = 4
            line_mask |= c_mask & np.less(distance, line_width)
            max_distance = np.maximum(max_distance, distance + 128)

            line_data[:, :, 3] = line_mask * 255
            max_distance = np.maximum(max_distance, np.zeros(max_distance.shape, dtype=float))
            max_distance = np.minimum(max_distance, np.zeros(max_distance.shape, dtype=float) + 255)
            seg_data[:, :, 3] = max_distance

    return {
        'prediction':class_data,
        'line_data': line_data,
        'fill_data': fill_data,
        'seg_data' : seg_data,
    }
    

def drawsegcontour( image, label, color=[255,0,0], line_width=4 ):
    '''
    Draw segmentation countour
    '''

    fill_data = np.ndarray((label.shape[0], label.shape[1], 4), dtype='uint8')
    for x in range(3):
        fill_data[:, :, x] = label.copy()

    # Assuming that class 0 is the background
    mask = np.greater(label, 0)
    fill_data[:, :, 3] = mask * 255
    line_data = fill_data.copy()
    seg_data = fill_data.copy()
    
    # Black mask of non-segmented pixels
    mask_data = np.zeros(fill_data.shape, dtype='uint8')
    mask_data[:, :, 3] = (1 - mask) * 255
            
    # Assuming that class 0 is the background.
    line_mask = np.zeros(label.shape, dtype=bool)
    max_distance = np.zeros(label.shape, dtype=float) + 1
    
    c_mask = np.equal(label, 1)
    
    # Find the signed distance from the zero contour
    distance = skfmm.distance(c_mask.astype('float32') - 0.5)
    # Accumulate the mask for all classes
    #line_width = 4
    line_mask |= c_mask & np.less(distance, line_width)
    max_distance = np.maximum(max_distance, distance + 128)

    line_data[:, :, 3] = line_mask * 255
    max_distance = np.maximum(max_distance, np.zeros(max_distance.shape, dtype=float))
    max_distance = np.minimum(max_distance, np.zeros(max_distance.shape, dtype=float) + 255)
    seg_data[:, :, 3] = max_distance


    #return {
    #    'prediction':class_data,
    #    'line_data': line_data,
    #    'fill_data': fill_data,
    #    'seg_data' : seg_data,
    #}

    image_sh = image.copy()
    image_sh[line_mask,:] = color;
    return image_sh;


def crf( image, label):
    '''
    Condicional random field
    '''
    
    n_label   = label.shape[1]
    im_height = label.shape[3]
    im_width  = label.shape[2]

    d = dcrf.DenseCRF2D(im_height, im_width, n_label)
    U = label.reshape((n_label,-1)) # Needs to be flat..reshape((n_label, im_height*im_width))
    Up = (U+0.001) / (np.sum(U, axis=0))
    
    img_np = image
    img_np = img_np.copy(order='C')

    # change to negative log probability for numerical reasons
    d.setUnaryEnergy(-np.log(Up))
    # gaussian pairwise potential
    d.addPairwiseGaussian(3, 1.5) # sigma_xy, comp # 3 1.5
    # bilateral pairwise potential
    d.addPairwiseBilateral(60, 10, img_np, 3)  # 60 10 3
    Q = d.inference(10)
    Q = np.array(Q).reshape((n_label, im_width,im_height))
    out_label = np.argmax(Q, axis=0)
    
    return out_label






# # def getweightmap(truth, w0=10, sigma=5):
# #     """
# #     Compute weights for pixels.
# #     Note: Copied from Siddhats code
# #     """

# #     truth_copy = truth.copy()
# #     truth_copy[truth == 2] = 1
# #     d1_f = ndimage.distance_transform_edt(truth_copy)
# #     truth_copy[truth == 0] = 1
# #     truth_copy[truth == 1] = 0
# #     d1_b = ndimage.distance_transform_edt(1 - truth)
# #     d1 = np.maximum(d1_f, d1_b)

# #     truth_copy = truth.copy()
# #     truth_copy[truth == 2] = 0

# #     labeled_array, num_features = ndimage.measurements.label(truth_copy, structure=None, output=None)

# #     tmp = np.empty_like(labeled_array)
# #     tmp[:] = labeled_array

# #     d2 = np.empty(labeled_array.shape)
# #     w_c = np.empty(labeled_array.shape)
# #     d2[:] = np.inf;
# #     w_c[:] = 0

# #     for ii in range(1, num_features + 1):
# #         tmp[:] = labeled_array
# #         tmp[np.logical_or(tmp == ii, tmp == 0)] = -1  # fg
# #         tmp[tmp != -1] = 0
# #         tmp[tmp == -1] = 1
# #         # everything except current class and bg goes to bg, current class and bg goes to fg
# #         d2_ii = ndimage.distance_transform_edt(tmp)
# #         d2[labeled_array == ii] = d2_ii[labeled_array == ii]

# #     tmp[:] = 1
# #     d2b1 = np.empty(labeled_array.shape)
# #     d2b2 = np.empty(labeled_array.shape)
# #     d2b1[:] = np.inf
# #     d2b2[:] = np.inf
# #     for ii in range(1, num_features + 1):
# #         tmp[labeled_array == ii] = 0
# #         db_ii = ndimage.distance_transform_edt(tmp)
# #         d2b1[labeled_array == 0], d2b2[labeled_array == 0] = np.minimum(db_ii[labeled_array == 0],
# #                                                                         d2b1[labeled_array == 0]), \
# #                                                              np.minimum(np.maximum(db_ii[labeled_array == 0],
# #                                                                                    d2b1[labeled_array == 0]),
# #                                                                         d2b2[labeled_array == 0])
# #         tmp[labeled_array == ii] = 1

   

# #     d2[labeled_array == 0] = d2b2[labeled_array == 0];
# #     # Calculate w_b
# #     w_b = w0 * np.exp(-np.square(d1+d2)/(2*sigma**2)); 


# #     # TODO produce waring or error when the image contain only a single class?

# #     assert truth.size > 0

# #     frac0 = np.sum(truth == 0) / float(truth.size)
# #     frac1 = np.sum(truth == 1) / float(truth.size)

# #     assert frac0 > 0
# #     assert frac1 > 0

# #     # Calculate w_c
# #     w_c[truth == 0] = 0.5 / (frac0)
# #     w_c[truth == 1] = 0.5 / (frac1)

    
# #     return w_c, w_b



