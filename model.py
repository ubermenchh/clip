import torch 
import torch.nn as nn 

from collections import OrderedDict 
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model*4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model*4, d_model))
            ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask 

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None 
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers 
        self.resblocks = nn.Sequential(*[AttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class ViT(nn.Module):
    def __init__(self, input_res, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_res = input_res 
        self.output_dim = output_dim 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding 
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj 

        return x


class CLIP(nn.Module):
    def __init__(self, embed_dim, image_res, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers):
        super().__init__()
        self.context_length = context_length 
        vision_heads = vision_width // 64
        self.visual = ViT(
                input_res=image_res,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
                )
        self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
                )
        self.vocab_size = vocab_size 
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)

        self.text_proj = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_sclae = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.init_params()

    def init_params(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-.5) * ((2 * self.transformer.layers)**.5)
        attn_std = self.transformer.width**-.5
        fc_std = (2 * self.transformer.width)**-.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init_normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_proj is not None:
            nn.init.normal_(self.text_proj, std=self.transformer.width**-.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask 
    
    def encode_image_and_text(self, image, text):
        x = self.token_embedding(text)

        x = x + self.positional_embedding 
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_proj 

        return self.visual(image), x 

    def forward(self, image, text):
        image_feat, text_feat = self.encode_image_and_text(image, text)

        image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        
        return image_feat, text_feat
