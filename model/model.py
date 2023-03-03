import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip import clip

from collections import OrderedDict
import os

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_proj = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text): 
        eot_idx = text.argmax(dim=-1)
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # BLD -> LBD
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.ln_final(x).type(self.dtype)
        x = torch.matmul(x, self.text_proj)     # [B, L, 512] -> [B, L, 512]

        x1 = x.clone()
        # gather_idx = mask_tensor(torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0]), eot_idx)
        # gather_idx = torch.cat((eot_idx, gather_idx), dim=1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x[torch.arange(x.shape[0]), eot_idx] = x1[:, 0]
        x[:, 0] = x1[torch.arange(x.shape[0]), eot_idx]

        # take global features from the eot embedding (eot_token is the highest number in each sequence)
        # return mask_tensor(x, eot_idx), x[torch.arange(x.shape[0]), eot_idx]
        return x[:, 1:, :], x[:, 0, :]
    
class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.patch_emb = clip_model.visual.conv1
        self.cls_emb = clip_model.visual.class_embedding
        self.pos_emb = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.image_proj = clip_model.visual.proj

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)   # [B, C, H, W] -> [B, D, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, D, H * W]
        x = x.permute(0, 2, 1)  # [B, H * W, D]
        x = torch.cat([self.cls_emb + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # [B, H * W + 1, D]
        x = x + self.pos_emb
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [B, L, D] -> [L, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [L, B, D] -> [B, L, D]

        x = self.ln_post(x)
        x = torch.matmul(x, self.image_proj)    # [B, L, 768] -> [B, L, 512]

        return x[:, 1:, :], x[:, 0, :]


class TokenCLIP(nn.Module):
    """ CLIP model that return all token features in forward() """
    def __init__(self, cfg):
        super().__init__()
        clip_model = load_clip_to_cpu(cfg)
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.embed_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        # B1 = B2
        # [B1, L1, D], [B1, D]
        patch_features, image_features = self.image_encoder(image.type(self.dtype))
        # [B2, L2, D], [B2, D]
        word_features, text_features = self.text_encoder(text)

        return patch_features, image_features, word_features, text_features
    
class SimCLIP(nn.Module):
    """ CLIP model that return similarities in forward() """
    def __init__(self, cfg, clip_model=None) -> None:
        super().__init__()
        if clip_model is not None:
            self.encoder = clip_model
        else:
            self.encoder = TokenCLIP(cfg)
        self.lamb = cfg.MODEL.LAMB


    def forward(self, image: torch.Tensor, text: torch.Tensor):
        patch_features, image_features, word_features, text_features = self.encoder(image, text)
        # Compute Similarity
        sim_g = image_features @ text_features.t()  # [B1, B2]
        c = torch.matmul(patch_features.unsqueeze(1), word_features.permute(0, 2, 1))  # [B1, B2, L1, L2]
        d = torch.matmul(patch_features, text_features.t()).permute((0, 2, 1))   # [B1, B2, L1]
        e = torch.matmul(word_features, image_features.t()).permute((2, 0, 1))   # [B1, B2, L2]
        emc = torch.mul(c, e.unsqueeze(-2))     # [B1, B2, L1, L2]
        dmc = torch.mul(c, d.unsqueeze(-1))     # [B1, B2, L1, L2]
        T_v = torch.mul(d.unsqueeze(-1), F.log_softmax(emc * self.lamb, dim=-1)) / d.shape[-1] # [B1, B2, L1, L2]
        T_t = torch.mul(e.unsqueeze(-2), F.log_softmax(dmc * self.lamb, dim=-2)) / e.shape[-1] # [B1, B2, L1, L2]
        # emc = torch.matmul(c.permute(1, 0, 2, 3), e.unsqueeze(-1)).squeeze(-1) # [B2, B1, L1]
        # dmc = torch.matmul(c.permute(0, 1, 3, 2), d.unsqueeze(-1)).squeeze(-1) # [B1, B2, L2]
        # T_v = torch.mul(d, F.softmax(emc.permute((1, 0, 2)) * self.lamb)) / e.shape[-1] # [B1, B2, L1]
        # T_t = torch.mul(e, F.softmax(dmc.permute((1, 0, 2)) * self.lamb)) / d.shape[-1] # [B2, B1, L2]

        sim_v = torch.mul(c, T_v).sum(dim=(2, 3))       # [B1, B2]
        sim_t = torch.mul(c, T_t).sum(dim=(2, 3)).t()   # [B2, B1]

        return sim_g, sim_v, sim_t

        


class ClsCLIP(nn.Module):
    """ CLIP model that fine tune for classification tasks """
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.encoder = clip_model
        self.dtype = clip_model.dtype
        self.classifier = nn.Sequential(OrderedDict([
                            ('ln', nn.LayerNorm(clip_model.embed_dim)), 
                            ('linear1', nn.Linear(clip_model.embed_dim, cfg.MODEL.HIDDEN_DIM)),
                            ('act', nn.ReLU(inplace=True)),
                            ('linear2', nn.Linear(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.CLASS_NUM)),
                            ]))

    def forward(self, image, text):
        # TODO:
        patch_features, image_features, word_features, text_features = self.encoder(image, text)
        x = torch.mul(image_features, text_features)    # [B, D]
        x = self.classifier(x)

        return x



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE_NAME
    if cfg.MODEL.PRETRAIN_FILE == "":
        # download file
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    else:
        # load local file
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.PRETRAIN_FILE)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            

def mask_tensor(tensor, indices):
    """ Mask a tensor at dim 1"""
    mask = torch.ones((tensor.shape[0], tensor.shape[1]), dtype=torch.bool)
    mask[torch.arange(tensor.shape[0]), indices] = False
    mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(tensor)
    return tensor[mask].reshape((tensor.shape[0], tensor.shape[1]-1, tensor.shape[2], tensor.shape[3]))

