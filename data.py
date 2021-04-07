from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
import spacy
from utils import printc
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import os
import json

spacy_eng = spacy.load("en_core_web_sm")
FOLDER_OF_PROJECT = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
class COCO(Dataset):
    def __init__(self, root, annFile,filepath,transform=None, freq_threadshold=5, train=True):
        self.data = CocoCaptions(
            root=root,
            annFile=annFile,
            transform=transform
        )
        self.vocab = Vocabulary(freq_threadshold)
        self.vocab.build_vocabulary(self.data, filepath)
        self.train = train

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, caption = self.data[index]
        if self.train:
            caption = caption[0]

            numericalized_caption = [int(self.vocab.stoi["<SOS>"])]
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption.append(int(self.vocab.stoi["<EOS>"]))


        else:
            numericalized_caption = [torch.tensor([int(self.vocab.stoi["<SOS>"])] + self.vocab.numericalize(item[1][0]) + [int(self.vocab.stoi["<EOS>"])]).long() for item in batch]

        return img, torch.tensor(numericalized_caption)





class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=int(self.pad_idx))

        return imgs, targets

class Vocabulary:
    def __init__(self, freq_threadshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = dict({v:int(i) for i,v in self.itos.items()})
        self.freq_threadshold = freq_threadshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, dataset, filepath):
        if not os.path.exists(filepath):
            printc("start build vocabulary")
            frequencies = {}
            idx = 4
            for _, sentence_list in tqdm(dataset, total=len(dataset)):
                for sentence in sentence_list:
                    for word in self.tokenizer_eng(sentence):
                        if word not in frequencies.keys():
                            frequencies[word] = 1
                        
                        else:
                            frequencies[word] += 1

                        if frequencies[word] == self.freq_threadshold:
                            self.stoi[word] = idx
                            self.itos[idx] = word
                            idx +=1

            self.save_vocab(filepath)
        else:
            self.load_vocab(filepath)

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            int(self.stoi[token]) if token in self.stoi.keys() else int(self.stoi["<UNK>"]) for token in tokenized_text
        ]


    def save_vocab(self,filepath):
        with open(filepath, 'w') as f:
            json.dump(self.itos, f)

    def load_vocab(self,filepath):
        with open(filepath, 'r') as f:
            self.itos = json.load(f)
            self.stoi = dict({v:i for i,v in self.itos.items()})

    def __len__(self):
        return len(self.stoi.keys())


#FOLDER_OF_PROJECT os.path.join(FOLDER_OF_PROJECT, "/dataset/COCO/captions/annotations/captions_train2014.json")
def get_loader(
        transform,
        freq_threadshold,
        filepath,
        dataset_folder,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    trainData = COCO(
        root=os.path.join(dataset_folder, "train2014"),
        annFile=os.path.join(dataset_folder, "annotations", "captions_train2014.json"),
        transform=transform,
        filepath=filepath,
        freq_threadshold=freq_threadshold,
        train=True,
    
    )
    testData = COCO(
        root=os.path.join(dataset_folder, "train2014"),
        annFile=os.path.join(dataset_folder, "annotations", "captions_train2014.json"),
        transform=transform,
        filepath=filepath,
        freq_threadshold=freq_threadshold,
        train=False,
    )
    pad_idx = trainData.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=trainData,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    test_loader = DataLoader(
        dataset=testData,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return train_loader, test_loader, trainData.vocab



if __name__ == "__main__":
    train_loader, _, _ = get_loader(transform, 5, 'vocab.json')
    for id, (img, caption) in enumerate(train_loader):
        print(f"id:{id}  caption{caption.shape}")
        if id == 1:
            break

