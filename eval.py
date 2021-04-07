import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from data_loader import get_loader
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from Img2Text import Img2Text
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

#TODO: utiliser un dataset autre que celui d'entreinement!

def eval(nb_test):
    train_loader, dataset = get_loader("dataset/flickr8k/images/", annotation_file="dataset/flickr8k/captions.txt", transform=transform, num_workers=2)
    loop = tqdm(enumerate(train_loader), total=nb_test, leave=False)

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1

    model = Img2Text(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load("my_checkpoint.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

    fig = plt.figure(figsize=(10,10))
    model.eval()
    with torch.no_grad():
        for idx, (imgs, captions) in loop:
            if idx+1 == nb_test:
                break
            
            ax = fig.add_subplot(2,2,idx+1)
            predicted_str, predicted_int = model.caption_image(imgs.to(device), dataset.vocab)
            #[dataset.vocab.itos[idx] for idx in result_caption]
            captions = [dataset.vocab.itos[idx] for idx in captions.squeeze(-1).tolist()]

            score = bleu_score([predicted_str[1:-1]], [captions])
            ax.imshow(imgs.squeeze(0).permute(1,2,0))
            text = f"CORRECT:{captions[1:-1]}\nPREDICTED:{predicted_str[1:-1]}\nBleu score:{score}"
            ax.title.set_text(text)
    plt.show()

if __name__=="__main__":
    eval(5)
        



