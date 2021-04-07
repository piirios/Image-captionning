import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json

def printc(text):
    print(f'[\33[32m>\33[0m] {text}')

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


class checkpointManager:
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(self.filepath):
            self.state = torch.load(self.filepath)
        else:
            #if os.path.isfile(self.filepath):
            folder = os.path.dirname(self.filepath)
            print(folder)
            os.makedirs(folder, exist_ok=True)
            self.state = None

            #else:
                #raise ValueError('filepath need a file into')

    def resume(self, model, optimizer):
        if self.state is None:
            print("return 0")
            return 0 #step = 0
        else:
            model.load_state_dict(self.state["state_dict"])
            optimizer.load_state_dict(self.state["optimizer"])
            return self.state["step"]

    def update(self, model,optimizer,step):
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
        
        if os.path.exists(self.filepath):
            if self.state is not None and checkpoint["step"] >= self.state["step"]:
                self.state = checkpoint
                torch.save(checkpoint, self.filepath)

        else:
            self.state = checkpoint
            torch.save(checkpoint, self.filepath)

    def save_model_params(self, filename, **params):
        data = {i:j for i,j in params.items()}
        with open(filename, 'w') as f:
            json.dump(data, filename)

    def load_model_params(self, filename):
        with open(filename, 'w') as f:
            data = json.load(f)
            return data


    
