import click
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import os
import json
import torchvision.transforms as transforms
from PIL import Image
from Img2Text import Img2Text
from data import get_loader
from pathlib import Path
from tqdm import tqdm
from utils import checkpointManager

CONF_FILE ="conf.json"
FOLDER_OF_PROJECT = os.path.dirname(os.path.abspath(__file__))

spacy_eng = spacy.load("en_core_web_sm")

transform = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.RandomCrop((299,299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_production = transforms.Compose([
    transforms.Resize((356,356)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def read_conf():
    with open(os.path.join(FOLDER_OF_PROJECT, CONF_FILE), 'r') as f:
        j = json.load(f)
        j['weight_folder'] = os.sep.join(j['weight_folder'].split(j['sep']))
        j['dataset_folder'] = os.sep.join(j['dataset_folder'].split(j['sep']))
        return j

def write_conf(data):
    data['sep'] = os.sep
    with open(os.path.join(FOLDER_OF_PROJECT, CONF_FILE), 'w') as f:
        json.dump(data, f)


@click.group()
def ic():
    pass

def save_model_params(filename, **params):
    data = {i:j for i,j in params.items()}
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_model_params(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data

conf = read_conf()

@ic.command()
@click.argument('folder', type=click.Path(exists=True))
def set_dataset_folder(folder):
    conf['dataset_folder'] = folder
    write_conf(conf)

@ic.command()
@click.argument('folder', type=click.Path(exists=True))
def set_weight_folder(folder):
    conf['weight_folder'] = os.path.abspath(folder)
    write_conf(conf)

@ic.command()
@click.option('--image','-img', type=click.Path(exists=True), required=True)
def caption(image):
    conf = read_conf()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, vocab = get_loader(transform, 5, 'vocab.json', conf['dataset_folder'])
    im = Image.open(image)
    if im.mode != 'RGB':
        #im.convert('RGB')
        bg = pure_pil_alpha_to_color_v2(im)
    else:
        bg = im
    bg = transform_production(bg)
    weight_file_path = os.path.join(Path(conf['weight_folder']), "weight.pth")
    conf_file_path = os.path.join(conf['weight_folder'], "model.conf")

    model = Img2Text(*load_model_params(conf_file_path).values()).to(device)
    state = torch.load(weight_file_path)
    model.load_state_dict(state['state_dict'])
    sq, _ = model.caption_image(torch.unsqueeze(bg, 0), vocab)
    click.echo(' '.join(sq))

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


"""
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 20
"""

def input_of_retrain():
    result = input(f" weight exist of the Img2Text model, do you want to retrain it? [Y/N]")
    if result == "Y" or result == "y":
        return True
    elif result=="N" or result=="n":
        return False
    else:
        print("please respond with Y or N")
        input_of_retrain()

@ic.command()
@click.option('--embed-size', default=256, type=int)
@click.option('--hidden-size', default=256, type=int)
@click.option('--num-layers', default=1, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--num-epochs', default=256, type=int)
def train(embed_size, hidden_size, num_layers, learning_rate, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_file_path = os.path.join(conf['weight_folder'], "model.conf")

    train_loader, val_loader, vocab = get_loader(transform, 5, 'vocab.json', conf['dataset_folder'])
    if os.path.exists(os.path.join(conf['weight_folder'], "weight.pth")):
        r = input_of_retrain()
        if r:
            os.remove(os.path.join(conf['weight_folder'], f"weight.pth"))
    else:
        r = True

    if r:
        model = Img2Text(embed_size, hidden_size, len(vocab), num_layers)
        save_model_params(conf_file_path,
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=len(vocab),
            num_layers=num_layers
        )
    else:
        model = Img2Text(*load_model_params(conf_file_path).values())
    

    manager = checkpointManager(os.path.join(Path(conf['weight_folder']), "weight.pth"))

    criterion = nn.CrossEntropyLoss(ignore_index=int(vocab.stoi["<PAD>"]))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    step = manager.resume(model, optimizer)

    # model = train_model(model, os.path.join(Path(conf['weight_folder']), "weight.pth"), learning_rate, num_epochs, train_loader, val_loader, vocab)
    
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for idx, (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)


            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape([-1]))

            step +=1 
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss = loss.item())
        manager.update(model, optimizer, epoch+1) 
    

if __name__ == '__main__':
    ic()