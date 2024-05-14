from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import *

class CondLoader(Dataset):

    def __init__(self, 
                 annot_path: str,
                 img_dir: str,
                 gene_img_name_col: str,
                 cond_img_name_col: Optional[str] = None,
                 cond_vec_name_cols: Optional[list] = [],
                 
                 # pp
                 emb_dims = 768,
                 img_size=128,
                 batch_size=8,
                 shuffle = False,
                 padding = False,
                 reverse = False,
                 dtype = torch.float16,
                
                 # aug
                 ratio_aug = False,
                 ratio_range = [1., 0.2],
                 rotate_aug = False,
                 rotate_list = [0,90,180,270],
                 hflip_aug = False,
                 vflip_aug = False,
                ):

        # init
        self.annot_path = annot_path
        self.img_dir = img_dir
        self.gene_img_name_col = gene_img_name_col
        self.cond_img_name_col = cond_img_name_col
        self.cond_vec_name_cols = cond_vec_name_cols
        
        self.annot = pd.read_csv(self.annot_path) 
        if bool(self.cond_vec_name_cols):
            self.vec_scaler = self.get_scaler()
        
        # pp
        self.emb_dims = emb_dims
        self.dtype = dtype
        self.img_size = img_size 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reverse = reverse
        self.padding = padding
        
        # aug
        self.ratio_aug  = ratio_aug
        self.ratio_range = ratio_range
        self.rotate_aug = rotate_aug
        self.rotate_list = rotate_list
        self.hflip_aug = hflip_aug
        self.vflip_aug = vflip_aug

    def img_pp(self, img, img_cond=None):
        
        # black-white reverse. background should be black!
        if self.reverse:
            img = Image.eval(img, lambda x: 255 - x)
            
        # ratio aug
        if self.ratio_aug:
            trial=0
            while trial<100:
                trial+=1
                try:
                    m = self.ratio_range[0]
                    s = self.ratio_range[1]
                    ratio = np.random.normal(m,s)
                    img = ratio_img(img, ratio)
                    img_cond = ratio_img(img_cond, ratio)
                    break
                except:
                    pass
        
        # padding to image
        if self.padding: 
            img = pad_img(img, target_size=self.img_size)
            if img_cond is not None:
                img_cond = pad_img(img_cond, target_size=self.img_size)
                
        else:
            img = img.resize((self.img_size, self.img_size))
            if img_cond is not None:
                img_cond = img_cond.resize((self.img_size, self.img_size))
                
        # rotate aug
        arr = np.array(img).astype('float')
        if img_cond is not None:
            arr_cond = np.array(img_cond).astype('float')

        if self.rotate_aug:
            angle = int(np.random.choice(self.rotate_list))
            if angle != 0:
                arr = rotate_arr(arr, angle=angle)
                if img_cond is not None:
                    arr_cond = rotate_arr(arr_cond, angle=angle)
                
        # to tensor 
        x = torch.tensor(arr, dtype=torch.float16).unsqueeze(0)
        if img_cond is not None:
            x_cond = torch.tensor(arr_cond, dtype=torch.float16).unsqueeze(0)
        
        # flip aug
        if self.hflip_aug:
            hflip = T.RandomHorizontalFlip(p=1.0)
            x = hflip(x)
            if img_cond is not None:
                x_cond = hflip(x_cond)
        if self.vflip_aug:
            vflip = T.RandomVerticalFlip(p=1.0)
            x = vflip(x)
            if img_cond is not None:
                x_cond = vflip(x_cond)
        
        if img_cond is not None:
            return x, x_cond
        else:
            return x
            
    def get_scaler(self, path=None):
        if path is None:
            target_params = self.annot[self.cond_vec_name_cols]
            scaler = MinMaxScaler()
            results = scaler.fit_transform(target_params)
        else:
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
        
        return scaler
        
    def transfer_vec(self, vec, norm=True):
        vf = [vec[0]] # volume fraction always first
        if len(vec)!=1:
            if norm:
                df = pd.DataFrame([vec[1:]], columns = self.cond_vec_name_cols) 
                conds = self.vec_scaler.transform(df)
                conds = list(conds[0])
            else: conds = vec[1:]
            
            vf = vf + conds

        return vf
        
    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        items = self.annot.iloc[idx]

        # img
        img_name = items[self.gene_img_name_col]
        img_path = f'{self.img_dir}/{img_name}'
        img = Image.open(img_path).convert('L')

        # img conds
        if bool(self.cond_img_name_col):
            img_name = items[self.cond_img_name_col]
            img_path = f'{self.img_dir}/{img_name}'
            cond_img = Image.open(img_path).convert('L')
        
            ## img pp
            img, cond_img = self.img_pp(img, cond_img)
            cond_img = 16*16
        else:
            img = self.img_pp(img)
        
        # basic volume fraction
        vec_conds = [get_vf_from_img(img)]

        # add vector conditions
        if bool(self.cond_vec_name_cols):
            conds = list(items[self.cond_vec_name_cols])
            vec_conds += conds
        vec_conds = self.transfer_vec(vec_conds)
        vec_conds = torch.tensor(vec_conds, dtype = self.dtype)
        
        # norm
        img = img/ 127.5 - 1.

        if bool(self.cond_img_name_col):
            return  img, vec_conds, cond_img
        else:
            return img, vec_conds

    
    def get_ds(self):
        return  DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_batch(self, idx=0):
        ds = self.get_ds()
        for i, batch in enumerate(ds):
            if i == idx : break
        return batch