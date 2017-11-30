import os
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from skimage.color import label2rgb
from skimage import measure
from skimage import morphology
import matplotlib.pyplot as plt


from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

import pydensecrf.densecrf as dcrf

def metric_dicecoef(label, prediction):
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

def test(args):

    # Setup image
    print("Read Input Image from : {}".format(args.img_path) )
    #img = misc.imread(args.img_path)
    ind=int(args.img_path)
    #ind=0

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True,serious_transform=True)
    n_classes = loader.n_classes

#     img = img[:, :, ::-1]
#     img = img.astype(np.float64)
# #    img -= loader.mean
#     img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
#     img = img.astype(float) / 255.0
#     # NHWC -> NCWH
#     img = img.transpose(2, 0, 1) 
#     img = np.expand_dims(img, 0)
#     img = torch.from_numpy(img).float()
    img,lab = loader[ind]
    img = img.unsqueeze(0)

    # Setup Model
    model = torch.load(args.model_path)
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0))
    else:
        images = Variable(img)

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

    npimg=np.squeeze(img.numpy(),axis=0).transpose(1,2,0)
    npimg=npimg[:,:,(2,1,0)]
    w=npimg.shape
    w1=pred.shape
    offset=(w[0]-w1[0])//2
    npimg=npimg[offset:-offset,offset:-offset,:]
    label=lab.numpy()
    label=label[offset:-offset,offset:-offset]

    #conditional random field
    unary =np.array(F.softmax(outputs).data.cpu().numpy())
    n_label = unary.shape[1]
    im_height = unary.shape[2]
    im_width = unary.shape[3]
    d = dcrf.DenseCRF2D(im_height, im_width, n_label)
    U = unary.reshape((n_label,-1)) # Needs to be flat..reshape((n_label, im_height*im_width))
    Up = (U+0.001) / (np.sum(U, axis=0))
    img_np = (np.array(npimg)*255).astype('uint8')
    img_np =img_np.copy(order='C')
    # change to negative log probability for numerical reasons
    d.setUnaryEnergy(-np.log(Up))
    # gaussian pairwise potential
    d.addPairwiseGaussian(3, 1.5) # sigma_xy, comp # 3 1.5
    # bilateral pairwise potential
    d.addPairwiseBilateral(60, 10, img_np, 3)  # 60 10 3
    Q = d.inference(50)
    Q = np.array(Q).reshape((n_label, im_height,im_width))
    out_label = np.argmax(Q, axis=0)
    crfimage=loader.decode_segmap(out_label)

    #visualization
    decoded = loader.decode_segmap(pred)
    
    #save unet
    contours = measure.find_contours(pred, 0.5)
    contim=np.zeros_like(pred)
    for n, contour in enumerate(contours):
        contim[contour[:,0].astype('uint32'), contour[:, 1].astype('uint32')]=1
    
    contim=morphology.binary_dilation(contim)
    contour=np.where(contim==True)

    image_label_overlay = (npimg*255).astype('uint8')
    image_label_overlay[contour[0],contour[1],1]=1.0

    contours = measure.find_contours(label, 0.5)
    contim=np.zeros_like(label)
    for n, contour in enumerate(contours):
        contim[contour[:,0].astype('uint32'), contour[:, 1].astype('uint32')]=1
    
    contim=morphology.binary_dilation(contim)
    contour=np.where(contim==True)

    image_label_overlay = (npimg*255).astype('uint8')
    image_label_overlay[contour[0],contour[1],0]=1.0

    im_name=os.path.join(args.out_path,loader.dataprov.files[ind])
    misc.imsave(im_name, image_label_overlay)

    dice=metric_dicecoef( label, pred)
    args.mean+=dice
    print('Unet:',dice,' Mean: ',args.mean/(ind+1))

    #save crf
    contours = measure.find_contours(out_label, 0.5)
    contim=np.zeros_like(out_label)
    for n, contour in enumerate(contours):
        contim[contour[:,0].astype('uint32'), contour[:, 1].astype('uint32')]=1
    
    contim=morphology.binary_dilation(contim)
    contour=np.where(contim==True)
    image_label_overlay = (npimg*255).astype('uint8')
    image_label_overlay[contour[0],contour[1],1]=1.0

    im_name=os.path.join(args.out_path,'crf',loader.dataprov.files[ind])
    misc.imsave(im_name, image_label_overlay)

    #print('Unet+CRF:',metric_dicecoef(label,out_label))
    #print("Segmentation Mask Saved at: {}".format(args.out_path) )

    ind+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()

    args.mean=0
    for i in range(1000):
        args.img_path=i
        test(args)
