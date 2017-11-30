
import caffe
import cv2
import numpy as np

import skimage.color as skcolor
import skimage.util as skutl
import scipy.misc
import random

import lib.imageutl as imutl


class carLoaderLayer(caffe.Layer):    
    '''
    Car loader 
    '''
    def setup(self, bottom, top):
        # Check top shape
        if len(top) != 2:
                raise Exception("Need to define tops (data and label)")
        
        #Check bottom shape
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        #Read parameters
        params = eval(self.param_str)
        src_file = params["src_file"]
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"]
        
        #Reshape top
        top[0].reshape(self.batch_size, 3, self.im_shape, self.im_shape)    
        top[1].reshape(self.batch_size, 1, self.im_shape, self.im_shape)

        #Read source file
        #I'm just assuming we have this method that reads the source file
        #and returns a list of tuples in the form of (img, label)
        #self.imgTuples = readSrcFile(src_file) 
        self.data = imutl.dataProvide( src_file, fn_image='train', fn_label='train_masks', lext='gif');

        
    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.load_next_image();
            
            #Preprocess the image
            im, label = self.preprocess(im, label, self.im_shape);
            im = im[np.newaxis,:,:].transpose(0,3,1,2).astype('float'); 
            label = label[np.newaxis, np.newaxis,:,:].astype('float');

            # Add directly to the top blob
            top[0].data[itt, ...] = im;
            top[1].data[itt, ...] = label;
    
    def load_next_image(self):
        #If we have finished forwarding all images, then an epoch has finished
        #and it is time to start a new one
        
        rand_idx = np.random.randint(self.data.num);
        im = self.data.getimage(rand_idx);
        label = self.data.getlabel();
                        
        return im, label
    
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (img shape and batch size)
        """
        pass

    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass

  
    def preprocess(self, image, label, size=572 ):
    
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

        label[label==255] = 1;
        label[label==2] = 255;

        return image, label;



class augmentationLayer(caffe.Layer):
    """
    Data Augmentation
    """

    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """

        # Check top shape
        if len(top) != 2:
                raise Exception("Need to define tops (data and label)")

        #Read parameters
        params = eval(self.param_str)
        self.angle = params["angle"]
        self.translation = params["translation"]
        self.warp = params["warp"]


    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        top[1].reshape(bottom[1].data.shape[0], bottom[1].data.shape[1], bottom[1].data.shape[2], bottom[1].data.shape[3])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        images, labels = self.transform_blobs_image(bottom, self.angle, self.translation, self.warp);
        top[0].data[...] = images;
        top[1].data[...] = labels;


    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        pass

    def transform_blobs_image(self, blobs, angle, translation, warp):
        """
        Tranform blobs
        """
        images = blobs[0].data;
        labels = blobs[1].data;        
        for i in range(images.shape[0]):            
            image = images[i,...].transpose(1,2,0);
            label = labels[i,...].transpose(1,2,0);
            image, label = self.transform_image(image, label, angle, translation, warp);
            images[i,...] = image.transpose(2,0,1);
            labels[i,...] = label;
        return images, labels 

        

    def transform_image(self, image, label, angle, translation, warp):
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
        image = image.astype('uint8');
        label = label.astype('uint8');
        label[label == 255] = 0;

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

        label[label > 0] = 1;

        return image, label




class diceLayer(caffe.Layer):
    """
    A layer that calculates the Dice coefficient
    """
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute Dice coefficient.")
        # compute sum over all axes but the batch and channel axes
        self.sum_axes = tuple(range(1, bottom[0].data.ndim - 1))

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            raise Exception("Prediction must have twice the number of elements of the input.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        
        label = bottom[1].data[:,0,:,:]
        # compute prediction
        prediction = np.argmax(bottom[0].data, axis=1)   

        # area of predicted contour
        a_p = np.sum(prediction, axis=self.sum_axes)
        # area of contour in label
        a_l = np.sum(label, axis=self.sum_axes)
        # area of intersection
        a_pl = np.sum(prediction * label, axis=self.sum_axes)
        # dice coefficient
        dice_coeff = np.mean(2.*a_pl/(a_p + a_l))

        top[0].data[...] = dice_coeff

    def backward(self, top, propagate_down, bottom):
        pass