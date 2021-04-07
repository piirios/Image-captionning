# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:18:05 2021

@author: Louis
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, checkpointManager
from data import get_loader
from tqdm import tqdm
import matplotlib.pyplot as plt


#l'embedding permet de représenter des variables discretes en vecteurs continues
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN 
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) #on modifie la dernière couche par une couche linéaire
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.inception(images)
        
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True #permet de calculer le gradient sur les dernières couches fc, donc de ne pas entreiner tout le réseau
            else:
                param.requires_grad = self.train_CNN
                
        return self.dropout(self.relu(features))
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        
        return outputs
    
    
class Img2Text(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Img2Text, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.DecoderRNN(features, captions)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.DecoderRNN.lstm(x, states)
                output = self.DecoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.DecoderRNN.embed(predicted).unsqueeze(0)
                if vocabulary.itos[str(predicted.item())] == "<EOS>":
                    break
        return [vocabulary.itos[str(idx)] for idx in result_caption][1:-1], result_caption
    
    
    


