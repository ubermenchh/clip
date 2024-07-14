import torch
import torch.nn as nn 
import numpy as np

from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
from dataclasses import dataclass 
import os

@dataclass
class Config:
    clip_model="openai/clip-vit-base-patch32"
    text_model="gpt2"
    num_workers=2
    train_size=0.8
    valid_size=0.1
    epochs=100
    lr=3e-3
    k=0.33
    batch_size=32
    ep_len=4
    num_layers=6
    n_heads=16
    fwd_expansion=4
    max_len=40
    dropout=0.1
    weights_dir=os.path.join("weights")
    device="cuda" if torch.cuda.is_available() else "cpu"


class ImageEncoder(nn.Module):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.device = device 
        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(device)

    def forward(self, img):
        img = self.preprocessor(images=img, return_tensors="pt").to(self.device)
        img_feats = self.model(**img)
        return img_feats.pooler_output

class TextDecoder(nn.Module):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embed, attn_mask=None):
        text_feats = self.model(inputs_embeds=embed, attention_mask=attn_mask)
        return text_feats.logits

class ProjectionHead(nn.Module):
    def __init__(self, ep_len, num_layers, embed_size, n_heads, fwd_exp, dropout, device="cpu"):
        super().__init__()
        self.ep_len = ep_len
        self.embed_size = embed_size
        self.device = device 

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=n_heads,
                dim_feedforward=embed_size * fwd_exp,
                dropout=dropout,
                batch_first=True,
                device=device
            ),
            num_layers = num_layers
        ).to(self.device)
        self.proj = nn.Linear(embed_size, ep_len*embed_size).to(self.device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, img_embed, train_mode=False):
        x = self.transformer_encoder(img_embed)
        x = self.proj(x)
        x = x.view(*([-1, self.ep_len, self.embed_size] if train_mode else [self.ep_len, self.embed_size]))
        return x

class Net(nn.Module):
    def __init__(self, clip_model, text_model, ep_len, num_layers, n_heads, fwd_expansion, dropout,
                 max_len, device="cpu"):
        super().__init__()
        self.device = device 
        self.ep_len = ep_len 

        self.encoder = ImageEncoder(model=clip_model, device=device)
        self.proj_head = ProjectionHead(ep_len=ep_len, num_layers=num_layers,
                                   embed_size=self.encoder.model.config.hidden_size,
                                   n_heads=n_heads, fwd_exp=fwd_expansion, dropout=dropout,
                                   device=device)
        self.decoder = TextDecoder(model=text_model, device=device)

        assert self.encoder.model.config.hidden_size == self.decoder.model.config.n_embd, "Embedding size of models mismatch"
        self.max_len = max_len 
        self.loss_fn = nn.CrossEntropyLoss()
        self.freeze_layers()

    def freeze_layers(self):
        for p in [*list(self.encoder.parameters()), *list(self.decoder.parameters())[14:-14]]:
            p.requires_grad = False 

    def forward(self, img, temperature=1.0):
        if temperature <= 0.0: temperature = 1.0

        with torch.no_grad():
            img_embd = self.encoder(img)
            img_proj = self.proj_head(img_embd) # (ep_len, embed_size)

            sos_embd = self.decoder.model.transformer.wte(torch.tensor(self.decoder.tokenizer.bos_token_id).to(self.device))
            sos_embd = sos_embd.unsqueeze(0) # (1, embed_size)

            start_embd = torch.cat([sos_embd, img_proj], dim=0) # (ep_len + 1, embed_size)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.decoder.model.transformer.wte(torch.tensor(tokens).to(self.device))
                    emb = torch.cat([start_embd, tok_emb], dim=0)
                else:
                    emb = start_embd 
                
                pos_emb = self.decoder.model.transformer.wte(torch.arange(emb.shape[0]).to(self.device))
                emb += pos_emb 
                pred = self.decoder(emb)
                pred = torch.softmax(pred / temperature, dim=-1)
                
                _, pred = torch.max(pred, dim=1)

                last_token = pred[-1].item()
                tokens.append(last_token)

                if last_token == self.decoder.tokenizer.eos_token_id: break 
            
            decoded = self.decoder.tokenizer.decode(tokens[:-1])

            decoded = decoded.strip()
            decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens 

    def train_forward(self, img_emb, trg_cap, attn_mask):
        x, x_mask = trg_cap[:, :-1], attn_mask[:, :-1]
        y = trg_cap[:, 1:]

        img_proj = self.proj_head(img_emb, train_mode=True)
        text_emb = self.decoder.model.transformer.wte(x)

        x = torch.concat([img_proj, text_emb], dim=1)
        x_mask = torch.concat([torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1)

        pos_emb = self.decoder.model.transformer.wpe(torch.arange(x.shape[1]).to(self.decoder.device))
        pos_emb = pos_emb.expand_as(x)

        x += pos_emb 

        res = self.decoder(x, attn_mask=x_mask)
        res = torch.softmax(res, dim=2)

        loss = self.loss_fn(res[:, self.ep_len:, :].reshape(-1, res.shape[-1]), y.reshape(-1))
        return loss

if __name__=="__main__":
    config = Config()

    model = Net(clip_model=config.clip_model, text_model=config.text_model, ep_len=config.ep_len,
                num_layers=config.num_layers, n_heads=config.n_heads, 
                fwd_expansion=config.fwd_expansion, dropout=config.dropout, max_len=config.max_len)

    model.eval()
    res = model(torch.tensor(np.random.randn(3, 224, 224).astype(np.uint8)))
    print(res[0])

    model.train()
    N = 10 
    emb = model.decoder.model.config.n_embd 
    length = 20 

    loss = model.train_forward(torch.rand(N, emb).to(config.device), 
                               torch.randint(1, 50000, (N, length)).to(config.device),
                               attn_mask=torch.concat([torch.ones(N, length-3).to(config.device),
                                                       torch.zeros(N, 3).to(config.device)], dim=1))
    print(f"Loss: {loss}")

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
