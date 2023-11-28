import torch

def set_device():
    if torch.cuda.is_available():
        device='cuda:0'
        print('Current environment has an available GPU.')
    else:
        device='cpu'
        print('Current environment does not have an available GPU.')
        
    return device