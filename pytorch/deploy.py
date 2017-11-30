
import os
import sys
import argparse
import numpy as np
import pandas as pd

# torch
import torch
import visdom
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F

from tqdm import tqdm
from skimage.color import label2rgb
from skimage import measure
from skimage import morphology
import scipy.misc as misc
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores
import pydensecrf.densecrf as dcrf


# local
import imageutl as imutl
import visualizationutl as view
import netutility as  netutl



PATHDATASET = 'db'
NAMEDATASET = 'train/image'
METADATA = 'metadata.csv'
PATHMODEL = '.'
NAMEMODEL = 'unet_cars_1_68.pkl'
RLNAME = 'test_masks.csv'
NUMITER = 5
BCRF = True
bTEST = True



def process( net, image, bcrf = True ):

    image_in, asp, image_in_oz = netutl.unet_transform_image_size(image, size=388)
    image_proc = image_in[:, :, ::-1]
    image_proc = image_proc.astype(float)

    # NHWC -> NCHW
    image_proc = image_proc.transpose(2, 0, 1)
    image_proc = image_proc[np.newaxis,...]
    image_proc = torch.from_numpy(image_proc).float()

    if torch.cuda.is_available():
        net.cuda(0)
        images_torch = Variable(image_proc.cuda(0), volatile=True )
    else:
        images_torch = Variable(image_proc)

    # fordward
    outputs = net(images_torch)

    # crf
    if bcrf:
        pred = np.array(F.softmax(outputs).data.cpu().numpy())
        pred_1 =netutl.inv_transform_label_size(pred[0,0,:,:],image.shape, asp)
        pred_2 =netutl.inv_transform_label_size(pred[0,1,:,:],image.shape, asp)
        pred=np.concatenate((pred_1[np.newaxis,np.newaxis,:,:],pred_2[np.newaxis,np.newaxis,:,:]),axis=1).astype('float32')
        label = netutl.crf(image, pred)

    else:
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        label = netutl.inv_transform_image_size(pred, image.shape, asp)
    
    return label





def validation():
    pass

def test():

    bcrf = BCRF
    pathnamedataset = os.path.join(PATHDATASET, NAMEDATASET)
    pathnamemodel = os.path.join(PATHMODEL, NAMEMODEL)

    # data provide
    data = imutl.imageProvide( pathnamedataset, ext='jpg' )
    numiter = NUMITER #data.num;

    # Setup Model
    net = torch.load(pathnamemodel)
    net.eval()

    datarl = list();
    for i in range( numiter ):

        # read data
        image = data.getimage(i)
        
        # estimate
        label_hat = process( net, image, bcrf );
        datarl.append({ 
            'img': data.getimagename(), 
            'rle_mask': netutl.rle_encode(label_hat) 
            })

        if (i+1) % 1 == 0:
            print('iteration: {}'.format(i))

    return datarl


if __name__ == '__main__':
    
    rlfilename = os.path.join('.',RLNAME);
    
    if bTEST:
        datarl = test();
    else: 
        datarl, dice = validation();
        print('Mean dice: {}'.format(np.mean(dice)) )

    df = pd.DataFrame(datarl)
    df.to_csv(rlfilename, index=False, encoding='utf-8')
    
    print('dir: {}'.format(rlfilename))
    print('finish!!!')