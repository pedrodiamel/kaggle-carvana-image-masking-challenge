
import os
import numpy as np 
import csv
import caffe
import pandas as pd
import scipy.misc

import lib.imageutl as imutl
import lib.netutility as netutl

PATHDATASET = '../db'
NAMEDATASET = 'car'
METADATA = 'metadata.csv'
PATHMODEL = '../net'
NAMEMODELPROTO = 'deploy.prototxt'
NAMEMODELCAFFE = 'model.caffemodel'
RLNAME = 'rl.csv'
NUMITER = 10
BCRF = True
BGPU = False
bTEST = False

def process( net, image, bcrf = True ):

    im, asp = netutl.transform_image_size(image, net.blobs['data'].data.shape[-1])
    im_input = im[np.newaxis, :, :].transpose(0,3,1,2);

    net.blobs['data'].data[...] = im_input;
    net_output = net.forward();
    prediction = net_output[net_output.keys()[0]].astype('float32')
    
    if bcrf:
        prediction = netutl.crf(im, prediction);        
    else:    
        prediction = prediction[0,:,:,:];
        prediction = np.argmax(prediction, axis=0).astype('uint8')
        prediction = np.equal(prediction, 1).astype('uint8')    

    label = netutl.inv_transform_image_size(prediction, image.shape[0:2], asp)
    return label;


def validation():
    
    pathname = os.path.join(PATHDATASET, NAMEDATASET);
    pathmetadata = os.path.join(pathname, METADATA)
    modelproto = os.path.join(PATHMODEL,NAMEMODELPROTO);
    modelcaffe = os.path.join(PATHMODEL,NAMEMODELCAFFE);
    rlfilename = os.path.join('.',RLNAME);
    numiter = NUMITER;
    bcrf = BCRF;
    bgpu = BGPU;

    # data provide
    data = imutl.dataProvide( pathname, ext='jpg', fn_image='train', fn_label='train_masks', lext='gif')
    numiter = np.min( (numiter, data.num) );

    # net
    if bgpu:
        caffe.set_mode_cpu();
    else:
        caffe.set_mode_cpu();

    net = caffe.Net(modelproto, modelcaffe, caffe.TEST);

    
    dice = np.zeros((numiter,));
    datarl = list();

    for i in range( numiter ):

        # read data
        image = data.getimage(i)
        label = data.getlabel()
        label = (label/255.0).astype(np.uint8)

        # estimate
        label_hat = process( net, image, bcrf );
         
        # metric
        dice[i] = netutl.dicecoef(label, label_hat)
        datarl.append({ 
            'img': data.getimagename(), 
            'rle_mask': netutl.rle_encode(label_hat) 
            })
        
        if i % 10 == 0:
            print('iteration: {} {}'.format(i,dice[i]))
    
    return datarl, dice


def test():
    
    pathname = os.path.join(PATHDATASET, NAMEDATASET);
    pathmetadata = os.path.join(pathname, METADATA)
    modelproto = os.path.join(PATHMODEL,NAMEMODELPROTO);
    modelcaffe = os.path.join(PATHMODEL,NAMEMODELCAFFE);
    bcrf = BCRF;
    bgpu =  BGPU;
    
    # data provide
    data = imutl.imageProvide( pathname, ext='jpg' )
    numiter = NUMITER #data.num;

    # net
    if bgpu:
        caffe.set_mode_cpu();
    else:
        caffe.set_mode_cpu();

    net = caffe.Net(modelproto, modelcaffe, caffe.TEST);

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
    
    print('finish!!!')


