from __future__ import print_function, division, absolute_import, unicode_literals
import os
import click

from unet import unet
from unet import util
from lib  import imageutl as imutl


def create_training_path(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path

#@click.command()
#@click.option('--data_root', default="../db/car")
#@click.option('--output_path', default="./unet_trained")
#@click.option('--training_iters', default=32)
#@click.option('--epochs', default=100)
#@click.option('--restore', default=False)
#@click.option('--layers', default=5)
#@click.option('--features_root', default=64)
#def launch(data_root, output_path, training_iters, epochs, restore, layers, features_root):

def launch():

    data_root = "../db/car";
    output_path = "./unet_trained";
    training_iters = 32;
    epochs = 100;
    restore = False;
    layers = 5;
    features_root = 64;


    print("Using data from: %s"%data_root)
    data_provider = imutl.procProvideEx( data_root, fn_image='train', fn_label='train_masks', lext='gif')

    net = unet.Unet(channels=data_provider.channels, 
                    n_class=data_provider.n_class, 
                    layers=layers, 
                    features_root=features_root,
                    cost="cross_entropy",
                    cost_kwargs=dict(class_weights=1.0, regularizer=0.001),
                    )
    
    path = output_path if restore else create_training_path(output_path)
    
    trainer = unet.Trainer(net, optimizer="adam", batch_size=2, opt_kwargs=dict(learning_rate=1e-4))
    path = trainer.train(data_provider, path, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=2, 
                         restore=restore)
     
    #x_test, y_test = data_provider(1)
    #prediction = net.predict(path, x_test)
    #print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
    

if __name__ == '__main__':
    launch()
