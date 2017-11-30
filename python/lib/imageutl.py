

import os
import numpy as np
import PIL.Image
import scipy.misc

import cv2
import skimage.color as skcolor
import skimage.util as skutl
import random

class imageProvide(object):
    '''
    Management the image resources  
    '''

    def __init__(self, path, ext='jpg', fn_image=''):
        
        if os.path.isdir(path) is not True:
            raise ValueError('Path {} is not directory'.format(path))
        
        self.fn_image = fn_image;
        self.path = path;
        self.pathimage = os.path.join(path, fn_image);

        #self.files = os.listdir(self.pathimage);
        self.files = [ f for f in sorted(os.listdir(self.pathimage)) if f.split('.')[-1] == ext ];
        self.num = len(self.files);
        
        self.ext = ext;
        self.index = 0;

    def getimage(self, i):
        '''
        Get image i
        '''
        #check index
        if i<0 and i>self.num: raise ValueError('Index outside range');
        self.index = i;
        pathname = os.path.join(self.path,self.fn_image,self.files[i]);        
        return np.array(self._loadimage(pathname));

    def next(self):
        '''
        Get next image
        '''
        i = self.index;        
        pathname = os.path.join(self.pathimage, self.files[i]); 
        im = self._loadimage(pathname);
        self.index = (i + 1) % self.num;
        return np.array(im);

    def getimagename(self):
        '''
        Get current image name
        '''
        return self.files[self.index];

    def isempty(self): return self.num == 0;

    def _loadimage(self, pathname):
        '''
        Load image using pathname
        '''

        if os.path.exists(pathname):
            try:
                image = PIL.Image.open(pathname)
                image.load()
            except IOError as e:
                raise ValueError('IOError: Trying to load "%s": %s' % (pathname, e.message) ) 
        else:
            raise ValueError('"%s" not found' % pathname)


        if image.mode in ['L', 'RGB']:
            # No conversion necessary
            return image
        elif image.mode in ['1']:
            # Easy conversion to L
            return image.convert('L')
        elif image.mode in ['LA']:
            # Deal with transparencies
            new = PIL.Image.new('L', image.size, 255)
            new.paste(image, mask=image.convert('RGBA'))
            return new
        elif image.mode in ['CMYK', 'YCbCr']:
            # Easy conversion to RGB
            return image.convert('RGB')
        elif image.mode in ['P', 'RGBA']:
            # Deal with transparencies
            new = PIL.Image.new('RGB', image.size, (255, 255, 255))
            new.paste(image, mask=image.convert('RGBA'))
            return new
        else:
            raise ValueError('Image mode "%s" not supported' % image.mode);
        
        return image;


class dataProvide(imageProvide):
    '''
    Management dataset <images, labes>
    '''
    def __init__(self, path, ext='jpg', fn_image='images', fn_label='labels', posfix='_mask', lext='jpg'):
        super(dataProvide, self).__init__(path, ext, fn_image );
        self.fn_label = fn_label;
        self.posfix = posfix;
        self.lext = lext;
                
    def getlabel(self):
        '''
        Get current label
        '''
        i = self.index;
        name = self.files[i].split('.');
        pathname = os.path.join(self.path,self.fn_label,'{}{}.{}'.format(name[0],self.posfix, self.lext) );        
        label = np.array(self._loadimage(pathname));
        if label.ndim == 3: label = label[:,:,0];
        return label;


class procProvide(dataProvide):
    '''
    Management dataset <images, labes>
    '''

    channels = 3
    n_class = 2

    def __init__(
        self, 
        path, 
        ext='jpg', 
        fn_image='images', 
        fn_label='labels', 
        posfix='_mask', lext='jpg',
        a_min=None, a_max=None
        ):
        super(procProvide, self).__init__(path, ext, fn_image, fn_label, posfix, lext );
        
        self.a_min = a_min if a_min is not None else -np.inf;
        self.a_max = a_max if a_min is not None else np.inf;


    def _load_data_and_label(self):
        
        data, label = self._next_data()

        train_data, labels = self._post_process(data, label)  
        
        train_data = train_data.astype( np.float64 )
        labels = labels.astype( np.bool ) 
        
        train_data = self._process_data(train_data)
        labels = self._process_labels(labels)  
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]
    
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class)

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """

        data, labels = self._resize(data,labels, size=572);
        return data, labels


    def _resize(self, image, label, size=572 ):
    
        height, width, channels = image.shape;

        image = np.array(image).astype(np.uint8);
        label = np.array(label).astype(np.uint8);

        asp = float(height)/width;
        w = size; 
        h = int(w*asp);

        image_x = scipy.misc.imresize(image, (h,w), interp='bilinear');
        label_x = scipy.misc.imresize(label, (h,w), interp='nearest', mode='F');
        
        image = np.zeros((w,w,3));
        label = np.zeros((w,w));

        ini = int(round((w-h) / 2.0));
        image[ini:h+ini,:,:] = image_x;
        label[ini:ini+h,:] = label_x;

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)
        label[label>0] = 1;

        return image, label;
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
      

    def _next_data(self):        
        image  = self.getimage( np.random.choice( self.num ) )
        label = self.getlabel()    
        return image, label



class procProvideEx(dataProvide):
    '''
    Management dataset <images, labes>
    '''

    channels = 3
    n_class = 2

    def __init__(
        self, 
        path, 
        ext='jpg', 
        fn_image='images', 
        fn_label='labels', 
        posfix='_mask', lext='jpg',
        a_min=None, a_max=None,
        fn_weight='weight',
        wext='tif'
        ):
        super(procProvideEx, self).__init__(path, ext, fn_image, fn_label, posfix, lext );
        
        self.a_min = a_min if a_min is not None else -np.inf;
        self.a_max = a_max if a_min is not None else np.inf;
        self.wext = wext;
        self.fn_weight = fn_weight;

    
    def getweight(self):
        '''
        Get current weigth
        '''
        i = self.index;
        name = self.files[i].split('.');
        pathname = os.path.join( self.path,self.fn_weight,'{}.{}'.format(name[0], self.wext) );        
        weight = np.array(self._loadimage(pathname));
        return weight.astype( np.uint8 );

    
    def _load_data_and_label(self):
        
        data, label, weight = self._next_data()

        train_data, labels, weights = self._post_process(data, label, weight)  
        
        train_data = train_data.astype( np.float32 )
        weights = weights.astype( np.float32 )
        labels = labels.astype( np.bool ) 
        
        train_data = self._process_data(train_data)
        labels = self._process_labels(labels)  
        weights = self._process_weight(weights)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]
    
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class), weights.reshape(1, ny, nx, 1)

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _process_weight(self, weight):
        weights = (weight.astype( np.float32 ) / 255.0) * 3.0;
        #nx = weight.shape[1]
        #ny = weight.shape[0]
        #weights = np.zeros((ny, nx, self.n_class), dtype=np.float32)
        #weights[..., 0] = weight
        #weights[..., 1] = weight
        return weights;
    
    def _post_process(self, data, labels, weight):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        :param weight: the weight array
        """

        data, labels = self._resize(data,labels, size=572);
        weight = self._resize_weight(weight, size=572);
        
        angle = 0;
        translation = 0.2;
        data, labels, weight = self.transform_image(data, labels, weight, angle, translation)
        return data, labels, weight


    def transform_image(self, image, label, weight, angle, translation):
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
        image  = image.astype('uint8');
        label  = label.astype('uint8');
        weight = weight.astype('uint8');
        
        # Rotation
        #center = (width//2, height//2)
        #angle_rand = np.random.uniform(-angle, angle)
        #rotation_mat = cv2.getRotationMatrix2D(center, angle_rand, 1)
        #image = cv2.warpAffine(image, rotation_mat, (width, height))

        # Translation
        x_offset = translation * width * np.random.uniform(-1, 1)
        y_offset = translation * height * np.random.uniform(-1, 1)
        translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])

        image  = cv2.warpAffine( image, translation_mat, (width, height))
        label  = cv2.warpAffine( label, translation_mat, (width, height))
        weight = cv2.warpAffine(weight, translation_mat, (width, height))


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

        label[label > 0] = 1;
        weight[weight<10] = 10;
        return image, label, weight


    def _resize(self, image, label, size=572 ):
    
        height, width, channels = image.shape;

        image = np.array(image).astype(np.uint8);
        label = np.array(label).astype(np.uint8);

        asp = float(height)/width;
        w = size; 
        h = int(w*asp);

        image_x = scipy.misc.imresize(image, (h,w), interp='bilinear');
        #label_x = scipy.misc.imresize(label, (h,w), interp='nearest', mode='F');
        label_x = scipy.misc.imresize(label, (h,w), interp='bilinear');

        
        image = np.zeros((w,w,3));
        label = np.zeros((w,w));

        ini = int(round((w-h) / 2.0));
        image[ini:h+ini,:,:] = image_x;
        label[ini:ini+h,:] = label_x;

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)
        label[label>0] = 1;

        return image, label;



    def _resize_weight(self, weight, size=572 ):
        
        height, width = weight.shape;
        weight = np.array(weight).astype(np.uint8);
        
        asp = float(height)/width;
        w = size; 
        h = int(w*asp);

        weight_x = scipy.misc.imresize(weight, (h,w), interp='bilinear');         
        weight = np.ones((w,w))*10;
        
        ini = int(round((w-h) / 2.0));
        weight[ini:ini+h,:] = weight_x;

        weight = weight.astype(np.float32)        
        return weight;

  
    
    def __call__(self, n):
        train_data, labels, weights = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
        W = np.zeros((n, nx, ny, 1))
    
        X[0] = train_data
        Y[0] = labels
        W[0] = weights
        for i in range(1, n):
            train_data, labels, weights = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
            W[i] = weights
    
        return X, Y, W
      

    def _next_data(self):        
        image  = self.getimage( np.random.choice( self.num ) )
        label  = self.getlabel() 
        weight = self.getweight()   
        return image, label, weight



class kittiProvide(imageProvide):
    '''
    Management dataset <images, labes>
    '''
    def __init__(self, path, ext='jpg', fn_image='images', fn_label='labels', posfix='_mask'):
        super(kittiProvide, self).__init__(path, ext, fn_image );
        self.fn_label = fn_label;
        self.posfix = posfix;
                
    def getlabel(self):
        '''
        Get current label
        '''
        i = self.index;
        name = self.files[i].split('.');
        pathname = os.path.join(self.path,self.fn_label,'{}{}.{}'.format(name[0],self.posfix,'txt') );        
        
        # get labels 
        labels = list();
        with open(pathname, 'r') as f:
            for line in f:
                labels.append(line.split());       
        return labels;





def image_to_array(image,
                   channels=None):
    """
    Returns an image as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    """

    if channels not in [None, 1, 3, 4]:
        raise ValueError('unsupported number of channels: %s' % channels)

    if isinstance(image, PIL.Image.Image):
        # Convert image mode (channels)
        if channels is None:
            image_mode = image.mode
            if image_mode not in ['L', 'RGB', 'RGBA']:
                raise ValueError('unknown image mode "%s"' % image_mode)
        elif channels == 1:
            # 8-bit pixels, black and white
            image_mode = 'L'
        elif channels == 3:
            # 3x8-bit pixels, true color
            image_mode = 'RGB'
        elif channels == 4:
            # 4x8-bit pixels, true color with alpha
            image_mode = 'RGBA'
        if image.mode != image_mode:
            image = image.convert(image_mode)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [3, 4])):
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] in [3, 4]:
                    # color to grayscale. throw away alpha
                    image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 3).reshape(image.shape + (3,))
            elif image.shape[2] == 4:
                # throw away alpha
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 4:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 4).reshape(image.shape + (4,))
                image[:, :, 3] = 255
            elif image.shape[2] == 3:
                # add alpha
                image = np.append(image, np.zeros(image.shape[:2] + (1,), dtype='uint8'), axis=2)
                image[:, :, 3] = 255
            elif image.shape[2] != 4:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')

    return image






def resize_image(image, height, width,
                 channels=None,
                 resize_mode=None,
                 ):
    """
    Resizes an image and returns it as a np.array

    Arguments:
    image -- a PIL.Image or numpy.ndarray
    height -- height of new image
    width -- width of new image

    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    """

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    image = image_to_array(image, channels)

    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image

    # Resize
    interp = 'bilinear'

    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return scipy.misc.imresize(image, (height, width), interp=interp)
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width]
        else:
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height) / 2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width) / 2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

        return image
