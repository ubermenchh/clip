import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader 
import torch.optim as optim
from transformers import AutoTokenizer

from model import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

def nt_xent_loss(image_feat, text_feat, temp):
    out = torch.cat([image_feat, text_feat], dim=0)
    n_samples = len(out)

    cov = torch.mm(out, out.t().contiguous())
    similarity = torch.exp(cov / temp)
    
    mask = ~torch.eye(n_samples, device=similarity.device).bool()
    neg = similarity.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    pos = torch.exp(torch.sum(image_feat * text_feat, dim=1) / temp)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()
    return loss


def train(train_loader, epochs=50, freqs=500, temp=2.65):
    counter = 0
    for _ in range(epochs):
        for _, data in enumerate(train_loader):
            image, labels = data
            label_str = [f"an image of {train_data.classes[x]}" for x in labels]
            text_tokens = torch.tensor(tokenizer(label_str).input_ids)

            optimizer.zero_grad()

            image_feat, text_feat = clip_model(image.to(device), text_tokens.to(device))

            loss = nt_xent_loss(image_feat, text_feat, temp)
            loss.backward()

            optimizer.step()

            if counter % freqs == 0:
                print("Training Step {counter}: Loss {loss}")
            counter += 1


if __name__=="__main__":    
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
    vocab_size = tokenizer.vocab_size 

    clip_model = CLIP(
        embed_dim=512, image_res=32, vision_layers=4, vision_width=512, vision_patch_size=4,
        context_length=4, vocab_size=vocab_size, transformer_width=512, transformer_heads=4, transformer_layers=4
    ).to(device)
    optimizer = optim.Adam(clip_model.parameters(), lr=3e-4)

    train(train_loader, epochs=50, freqs=500, temp=2.65)
   
