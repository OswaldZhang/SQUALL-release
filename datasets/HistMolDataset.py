import os
import math
import json
import glob
import torch
import random
from skimage import io
import torch.utils.data as data
from .build import DATASETS
import pandas as pd
from utils.logger import *
import json
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np

@DATASETS.register_module()
class HistMol(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        print(self.data_root)
        self.subset = config.subset
        print(self.subset)
        if config.resolution_dict:
            self.resolution_list = json.load(open(os.path.join(self.data_root, config.resolution_dict)))
        self.label_dict = json.load(open(os.path.join(self.data_root,config.label_dict)))
        print(self.data_root)

        file_list = glob.glob(os.path.join(self.data_root, '*/*.pt'))#for lowres
        if len(file_list) == 0:
            file_list = glob.glob(os.path.join(self.data_root, '*/*/*.pt'))
        #file_list = glob.glob(os.path.join(self.data_root, '*/*/*.pt'))#for highres
        self.file_list = []
        if_tile = 0
        if "posX" in list(self.label_dict.keys())[0]:
            if_tile = 1
        sample_id_check = []
        for file in file_list:
            if if_tile:
                sample_id = file.split('/')[-1].split("_expr.pt")[0]
            else:
                sample_id = file.split('/')[-2]
            self.file_list.append((file, float(self.label_dict[sample_id]['label'])))
            '''
            if len(self.file_list)>= 4096:#debug
                break
            '''
            sample_id_check.append(sample_id)
        sample_id_check = list(set(sample_id_check))
        '''
        #This is slow, please clean all datas before training
        sample_id_not_found = []
        for i in self.label_dict.keys():
            if i not in sample_id_check:
                sample_id_not_found.append(i)
        '''
        print(self.file_list[:10])
        print_log(f'[DATASET histmol] {len(self.file_list)} tiles were loaded', logger='Dataset')
        print_log(f'[DATASET histmol] {len(sample_id_check)} samples were loaded', logger='Dataset')
        print('[DATASET histmol samples were not found:')
        #print(sample_id_not_found)
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        tif_sample = expr_sample.replace("_expr.pt", "_HE.tif")
        #tif_sample = expr_sample.replace("_expr.pt", "_HE.tif").replace("/mnt/data/histMol/binary","/mnt/data/histMol/binary_stain")
        #print("tif_sample",tif_sample)
        try:
            expr = torch.load(expr_sample, weights_only=True)
        except Exception as e:
            print(f"Error at idx: {idx}, file: {expr_sample}, error: {e}")
            raise e 
        #expr = torch.load(expr_sample, weights_only=True)
        rgb = io.imread(tif_sample)
        rgb = torch.from_numpy(rgb).float()
        #print("norm check rgb",rgb)

        #expr = expr.to_dense().float() #qbw10.16
        #size = int(math.sqrt(expr.shape[0]))
        #expr = expr.reshape(size, size, expr.shape[-1])#qbw10.16

        sample_id = expr_sample.split('/')[-2]
        if "posX" in list(self.label_dict.keys())[0]:
            res = torch.tensor(self.resolution_list[expr_sample.split('/')[-1].split("_expr.pt")[0]]["resolution"])#)
        else:
            res = torch.tensor(self.resolution_list[expr_sample.split('/')[-2]]["resolution"])#)
        label = torch.tensor(int(label), dtype=torch.long)

        return rgb, expr, res, label, sample_id

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class HistMolKather(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        print(self.data_root)
        self.subset = config.subset
        print(self.subset)
        self.resolution_list = json.load(open(os.path.join(self.data_root, config.resolution_dict)))
        #self.resolution_list = json.load(open(os.path.join(self.data_root, 'resolution.json')))
        #self.label_dict = json.load(open(os.path.join(self.data_root, 'tissue.json')))
        print(self.data_root)
        self.label_dict = {"ADI":0,"BACK":1, "DEB":2,"LYM":3,"MUC":4,"MUS":5,"NORM":6,"STR":7,"TUM":8}
        #file_list = glob.glob(os.path.join(self.data_root, '*/*.pt'))#for lowres
        file_list = glob.glob(os.path.join(self.data_root, '*/*.tif'))#for highres
        self.file_list = []
        if_tile = 0
        samples = []
        for file in file_list:
            self.file_list.append((file, self.label_dict[file.split("/")[-2]]))
            samples.append(file.split("/")[-1].split(".")[0])
        print_log(f'[DATASET] {len(self.file_list)} tiles were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        tif_sample = expr_sample
        rgb = io.imread(tif_sample)
        rgb = torch.from_numpy(rgb).float()

        sample_id = expr_sample.split("/")[-1].split(".")[0]
        pos = sample_id
        res = torch.tensor(0.5)
        label = torch.tensor(int(label), dtype=torch.long)

        return rgb,res, label, sample_id,pos

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class HistMolRGB(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        print(self.data_root)
        self.subset = config.subset
        print(self.subset)
        self.resolution_list = json.load(open(os.path.join(self.data_root, config.resolution_dict)))
        self.label_dict = json.load(open(os.path.join(self.data_root,config.label_dict)))
        #self.resolution_list = json.load(open(os.path.join(self.data_root, 'resolution.json')))
        #self.label_dict = json.load(open(os.path.join(self.data_root, 'tissue.json')))
        print(self.data_root)

        #file_list = glob.glob(os.path.join(self.data_root, '*/*.pt'))#for lowres
        file_list = glob.glob(os.path.join(self.data_root, '*/*/*.pt'))#for highres
        self.file_list = []
        if_tile = 0
        if "posX" in list(self.label_dict.keys())[0]:
            if_tile = 1
        samples_count = {}  #  sample_id 
        samples = []
        for file in file_list:
            if if_tile:
                sample_id = file.split('/')[-1].split("_expr.pt")[0]
            else:
                sample_id = file.split('/')[-2]
            if sample_id not in self.label_dict or sample_id not in self.resolution_list :
                continue
            if type(self.resolution_list[sample_id]) == dict:
                self.resolution_list[sample_id] = self.resolution_list[sample_id]["resolution"]
            if math.isnan(self.resolution_list[sample_id]):
                continue
            if self.label_dict[sample_id]['set'] == self.subset and self.label_dict[sample_id]['label'] !="-1":# add for human only
                if sample_id not in samples_count:
                    samples_count[sample_id] = 0
                self.file_list.append((file, self.label_dict[sample_id]['label']))
                samples.append(sample_id)
                samples_count[sample_id] += 1  # 
        print_log(f'[DATASET] {len(self.file_list)} tiles were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        tif_sample = expr_sample.replace("_expr.pt", "_HE.tif")
        rgb = io.imread(tif_sample)
        rgb = torch.from_numpy(rgb).float()
        pos = "_".join(expr_sample.split("/")[-1].split("_")[:-1])

        sample_id = expr_sample.split('/')[-2]
        if "posX" in list(self.label_dict.keys())[0]:
            #print(self.resolution_list[expr_sample.split('/')[-1].split("_expr.pt")[0]])
            res = torch.tensor(self.resolution_list[expr_sample.split('/')[-1].split("_expr.pt")[0]])#)
        else:
            res = torch.tensor(self.resolution_list[expr_sample.split('/')[-2]]["resolution"])#)
        label = torch.tensor(int(label), dtype=torch.long)

        return rgb,res, label, sample_id,pos

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class HistMolSlideEmbedding(data.Dataset):
    def __init__(self, config):
        print("USING HistMolSlideEmbedding")
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.sample_num = config.sample_num

        self.file_list = glob.glob(os.path.join(self.data_root, f'{self.subset}/*'))
        print_log(f'[DATASET] {len(self.file_list)} slides were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"
        print(self.file_list)

    def __getitem__(self, idx):
        data_path = self.file_list[idx]
        embedding = torch.load(os.path.join(data_path, 'embedding.pt'), map_location='cpu')
        embedding = torch.stack(random.choices(embedding, k=self.sample_num))
        label = torch.load(os.path.join(data_path, 'label.pt'), map_location='cpu')

        return embedding, label

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class TCGASlideEmbedding(data.Dataset):
    def __init__(self, config):
        print("USING TCGASlideEmbedding")
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.sample_num = config.sample_num
        self.label_dict = json.load(open(config.label_dict))
        #file_list = glob.glob(os.path.join(self.data_root, '*.pt'))
        def fast_glob(data_root, ext="pt"):
            file_list = []
            for root, _, files in os.walk(data_root):
                file_list.extend(os.path.join(root, f) for f in files if f.endswith(f".{ext}"))
            return file_list

        file_list = fast_glob(self.data_root, ext="pt")


        print("len(file_list)",len(file_list))
        print("len(label_dict)",len(list(self.label_dict.keys())))
        self.file_list = []
        samples_count = {}  #  sample_id 
        samples = []
        for file in file_list:
            sample_id = file.split('/')[-1].split(".pt")[0]
            if sample_id not in self.label_dict:
                #print(sample_id,"not in resolution dict or label dict")
                continue
            if self.label_dict[sample_id]['set'] == self.subset:
                if sample_id not in samples_count:
                    samples_count[sample_id] = 0

                #if samples_count[sample_id]<1024:
                if "status" in self.label_dict[sample_id].keys():
                    self.file_list.append((file, {"status":self.label_dict[sample_id]['status'],"time":self.label_dict[sample_id]['time']}))
                else:
                    self.file_list.append((file, self.label_dict[sample_id]['label']))
                samples.append(sample_id)
                samples_count[sample_id] += 1  # 
        print_log(f'[DATASET] {len(self.file_list)} slides were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        expr = torch.load(expr_sample, weights_only=True)
        pos = "origin"
        sample_id = expr_sample.split('/')[-1].split(".pt")[0]
        res = torch.tensor(0.5, dtype=torch.float32)
        #label = torch.tensor(int(label), dtype=torch.long)
        if type(label) == dict:#survival
            label = torch.tensor([label["time"], label["status"]], dtype=torch.float32)### qbw survival
        elif type(label) == int:#drug response
            label = torch.tensor(label,dtype=torch.float32)
        elif type(label) == str:#drug response
            label = torch.tensor(int(label),dtype=torch.float32)
        else:#RNA or CNV
            label = torch.from_numpy(np.array(label)).float()
        return expr,res, label, sample_id,pos

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class HistMolABMIL(data.Dataset):
    # like HistMolRGB
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        #self.resolution_list = json.load(open(os.path.join(self.data_root, config.resolution_dict)))
        self.label_dict = json.load(open(os.path.join(self.data_root,config.label_dict)))
        file_list = glob.glob(os.path.join(self.data_root, '*/*.pt'))
        self.file_list = []
        if_tile = 0
        if "posX" in list(self.label_dict.keys())[0]:
            if_tile = 1
        self.resolution_list = {}
        samples_count = {}  #  sample_id 
        samples = []
        for file in file_list:
            if if_tile:
                sample_id = file.split('/')[-1].split("_expr.pt")[0]
            else:
                sample_id = file.split('/')[-2]
            if sample_id not in self.label_dict:
                continue
            if "TCGA" in sample_id:
                self.resolution_list[sample_id] = 0.5
            elif type(self.resolution_list[sample_id]) == dict:
                self.resolution_list[sample_id] = self.resolution_list[sample_id]["resolution"]
            if self.label_dict[sample_id]['set'] == self.subset:
                if sample_id not in samples_count:
                    samples_count[sample_id] = 0

                if samples_count[sample_id]< 256:
                    if "status" in self.label_dict[sample_id].keys():
                        self.file_list.append((file, {"status":self.label_dict[sample_id]['status'],"time":self.label_dict[sample_id]['time']}))
                    else:
                        self.file_list.append((file, self.label_dict[sample_id]['label']))
                    samples.append(sample_id)
                    samples_count[sample_id] += 1  # 
        print_log(f'[DATASET] {len(self.file_list)} tiles were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        tif_sample = expr_sample.replace("_expr.pt", "_HE.tif")
        #tif_sample = expr_sample.replace("_expr.pt", "_HE.tif").replace("/mnt/data/histMol/binary","/mnt/histMol/lowres_data/binary_stain")
        #print("tif_sample",tif_sample)
        pos = "_".join(expr_sample.split("/")[-1].split("_")[:-1])
        rgb = io.imread(tif_sample)
        if rgb.max() > 1:  # Assuming range is 0-255
            rgb = torch.from_numpy(rgb).float() / 255.0  # Normalize to 0-1
        else:  # Already normalized
            rgb = torch.from_numpy(rgb).float()
        rgb = rgb.to(torch.bfloat16)# Add for test
        #for lab 
        #rgb = torch.load(expr_sample, weights_only=True)
        sample_id = expr_sample.split('/')[-2]
        res = torch.tensor(self.resolution_list[sample_id])
        #label = torch.tensor(int(label), dtype=torch.long)
        #label = torch.tensor([label["time"], label["status"]], dtype=torch.float32)### qbw survival

        
        if type(label) == dict:#survival
            label = torch.tensor([label["time"], label["status"]], dtype=torch.float32)### qbw survival
        elif type(label) == int:#drug response
            label = torch.tensor(label,dtype=torch.float32)
        elif type(label) == str:#drug response
            label = torch.tensor(int(label),dtype=torch.float32)
        else:#RNA or CNV
            label = torch.from_numpy(np.array(label)).float()
        
        return rgb,res, label, sample_id,pos

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class HistMolABMIL_embedding(data.Dataset):
    # like HistMolRGB
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        file_list = glob.glob(os.path.join(self.data_root, '*/*.tif'))
        '''
        def load_pt_file_list(data_root):
            file_list = []
            txt_file_path = os.path.join(data_root, "filename.txt")
            if os.path.isfile(txt_file_path):
                with open(txt_file_path, "r") as f:
                    file_list.extend(f.read().splitlines()) 
            print(f"Loaded {len(file_list)} .pt files from all filename.txt files")
            return file_list
        file_list = load_pt_file_list(self.data_root)
        '''
        self.resolution_list = {}
        self.file_list = []
        samples_count = {}  #  sample_id 
        samples = []
        for file in file_list:
            sample_id = file.split('/')[-2]
            if "TCGA" in sample_id:
                sample_id = "-".join(sample_id.split("-")[:3])
            self.resolution_list[sample_id] = 0.5
            if sample_id not in samples_count:
                samples_count[sample_id] = 0
            #if samples_count[sample_id]<1024:
            self.file_list.append((file, 0))
            samples.append(sample_id)
            samples_count[sample_id] += 1  # 
        print_log(f'[DATASET] {len(self.file_list)} tiles were loaded', logger='Dataset')
        assert len(self.file_list) > 0, "No dataset!"

    def __getitem__(self, idx):
        expr_sample, label = self.file_list[idx]
        #tif_sample = expr_sample.replace("_expr.pt", "_HE.tif")
        tif_sample = expr_sample
        #tif_sample = expr_sample.replace("_expr.pt", "_HE.tif").replace("/mnt/data/histMol/binary","/mnt/histMol/lowres_data/binary_stain")
        #print("tif_sample",tif_sample)
        pos = "_".join(expr_sample.split("/")[-1].split("_")[:-1])
        rgb = io.imread(tif_sample)
        if rgb.max() > 1:  # Assuming range is 0-255
            rgb = torch.from_numpy(rgb).float() / 255.0  # Normalize to 0-1
        else:  # Already normalized
            rgb = torch.from_numpy(rgb).float()

        #for lab 
        #rgb = torch.load(expr_sample, weights_only=True)
        sample_id = expr_sample.split('/')[-2]
        if "TCGA" in sample_id:
            sample_id = "-".join(sample_id.split("-")[:3])
        res = torch.tensor(self.resolution_list[sample_id])
        #label = torch.tensor(int(label), dtype=torch.long)
        if type(label) == dict:#survival
            label = torch.tensor([label["time"], label["status"]], dtype=torch.float32)### qbw survival
        elif type(label) == int:#drug response
            label = torch.tensor(label,dtype=torch.float32)
        elif type(label) == str:#drug response
            label = torch.tensor(int(label),dtype=torch.float32)
        else:#RNA or CNV
            label = torch.from_numpy(np.array(label)).float()
        return rgb,res, label, sample_id,pos

    def __len__(self):
        return len(self.file_list)
