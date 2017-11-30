import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import cv2
import skimage.color as skcolor
import skimage.util as skutl
from ptsemseg.loader.imageutl import *
import math

from torch.utils import data


class carsLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=[512,512],serious_transform=True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.serious_transform=serious_transform
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.dataprov = dataProvide( self.root, fn_image='image', fn_label='label', lext='gif')

        # for split in ["train", "test", "val"]:
        #     file_list = os.listdir(root + '/' + split + '/image')
        #     self.files[split] = file_list

    def __len__(self):
        return self.dataprov.num

    def __getitem__(self, index):

        img = self.dataprov.getimage(index)
        lbl = self.dataprov.getlabel()

        #lbl = np.repeat(lbl[:, :, np.newaxis], 3, axis=2)

        # img_name = self.files[self.split][index]
        # img_path = self.root + '/' + self.split + '/image/' + img_name
        # lbl_path = self.root + '/' + self.split + '/label/' + img_name.replace(".jpg","_mask.gif")

        # img = m.imread(img_path)
        # img = np.array(img, dtype=np.uint8)

        # lbl = m.imread(lbl_path)
        # lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform_size(img, lbl)
            if self.serious_transform==True:
                img, lbl = self.transform_image(img, lbl, angle=10, translation=0.2, warp=0)
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform_size(self, image, label, size=388):

        height, width, channels = image.shape;

        image = np.array(image)
        label = np.array(label)

        asp = float(height)/width;
        w = size; 
        h = int(w*asp);

        image_x = m.imresize(image.copy(), (h,w), interp='bilinear');
        label_x = m.imresize(label.copy(), (h,w), interp='nearest', mode='F');

        image = np.zeros((w,w,3));
        label = np.zeros((w,w));

        ini = int(round((w-h) / 2.0));
        image[ini:ini+h,:,:] = image_x;
        label[ini:ini+h,:] = label_x;


        #_,label = cv2.threshold(label,127,255,0);
        #_,contours,_ = cv2.findContours(label, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        #for cnt in contours: cv2.drawContours(label, cnt, -1, 2, 1)

        image = image/255.0
        label = label.astype(np.uint8)
        label = (label==255).astype(np.uint8);

        downsampleFactor = 16;
        d4a_size= 0;
        padInput =   (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2;
        padOutput = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2;
        d4a_size = math.ceil( (size - padOutput)/downsampleFactor);
        input_size = downsampleFactor*d4a_size + padInput;

        offset=(input_size-size)//2
        image_f = np.zeros((input_size,input_size,3));
        label_f = np.zeros((input_size,input_size));

        image_f[offset:-offset,offset:-offset,:]=image
        label_f[offset:-offset,offset:-offset]=label
        return image_f, label_f;


    def transform_image(self,image, label, angle, translation, warp):
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

        #print(np.unique(label))

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

        #image = (image*255).astype(np.uint8);
        #label = (label==255).astype(np.uint8);

        return image, label

    def transform(self, img, lbl):

        img = img[:, :, ::-1]
        lbl = np.array(lbl)     
        
        #img = img.astype(np.float64)
        #img -= self.mean
        img = img.astype(float)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        
        Car = [255, 0, 0]
        Unlabelled = [0, 0, 0]

        label_colours = np.array([Unlabelled,Car])
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        rgb=rgb.astype(np.uint8)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb



if __name__ == '__main__':
    
    local_path = '/data/projects/carseg/carsegmentation/pytorch-semseg-master/db/train/'
    dst = carsLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)

    for i, sample in enumerate(trainloader):
        imgs, labels = sample
        if i % 10:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[0]))
            plt.show()
