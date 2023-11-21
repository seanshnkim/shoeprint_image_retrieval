import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(img, text=None):
    npimg = img.cpu().numpy()
    plt.axis("off")
    if text:
        plt.text(75,8,text,style='italic', fontweight='bold',
             bbox={'facecolor':'white','alpha':0.8,'pad':10})
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
    

def set_device():
    if torch.cuda.is_available():
        device='cuda:0'
        print('Current environment has an available GPU.')
    else:
        device='cpu'
        print('Current environment does not have an available GPU.')
        
    return device