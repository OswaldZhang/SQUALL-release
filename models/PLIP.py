import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
from tqdm import tqdm
from typing import List, Union
from torch.utils.data import DataLoader
import PIL
from models.Storm import ABMIL,Attn_Net_Gated
from transformers import CLIPModel, CLIPProcessor
#from datasets import Dataset, Image
from models.Storm import ABMIL,Attn_Net_Gated

@MODELS.register_module()
class PLIP_ABMIL(ABMIL):
    def __init__(self,config ,local_ckpt_path=None):
        super().__init__(
            config
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("/lustre1/zxzeng/bwqin/STORM_main/ckpt_others/plip_model/", local_files_only=True)  # no download
        self.processor = CLIPProcessor.from_pretrained("/lustre1/zxzeng/bwqin/STORM_main/ckpt_others/plip_model/", local_files_only=True)
        self.model = self.model.to(self.device)
        #CLIP training
        for param in self.model.parameters():
            param.requires_grad = True  # gradient

        # qbw 11.14 gating attention
        fc = [nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU()]
        dropout = True
        gate = True
        if dropout:
            fc.append(nn.Dropout(0.25))
        
        if gate:
            attention_net = Attn_Net_Gated(L=self.embed_dim, D=int(self.embed_dim/4), dropout=dropout, n_classes=1)
        else:
            attention_net = nn.Linear(self.embed_dim, 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        #x : ( B , 784 , config.cls_dim )
        
        self.classifier = nn.Linear(self.embed_dim, self.cls_dim)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # init weight qzk
        self.apply(self._init_weights)
        self.build_loss_func()
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

    def forward(self, images,res):
        images = self.processor(images=images, return_tensors="pt").to(self.device)
        img_features = self.model.get_image_features(**images)  # (B, 1024)
        A, x = self.attention_net(img_features)
        A = torch.transpose(A, 1, 0)
        bag_feature = torch.mm(F.softmax(A, dim=1), x)
        attn_scores = A
        output = self.classifier(bag_feature)

        return output, attn_scores

    def load_model_from_ckpt(self, ckpt_path):
        pass #this time no loading ckpt

@MODELS.register_module()
class PLIP_getembedding(nn.Module):
    def __init__(self,config ,local_ckpt_path=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("/lustre1/zxzeng/bwqin/STORM_main/ckpt_others/plip_model/", local_files_only=True)  # no download
        self.processor = CLIPProcessor.from_pretrained("/lustre1/zxzeng/bwqin/STORM_main/ckpt_others/plip_model/", local_files_only=True)
        self.model = self.model.to(self.device)
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

    def forward_rgb(self, images,res):
        images = self.processor(images=images, return_tensors="pt",do_rescale=False).to(self.device)
        img_features = self.model.get_image_features(**images)  # (B, 1024)
        return img_features
    def load_model_from_ckpt(self, ckpt_path):
        pass #this time no loading ckpt
