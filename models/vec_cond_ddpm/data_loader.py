import glob, os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from utils import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn



class Loader(Dataset):
    def __init__( self, 
                  annot_path,
                  img_dir,
                  img_col = '',
                  vec_cols = [],
                  emb_dims = 768,
                  
                  # pp
                  img_size=128,
                  batch_size=8,
                  shuffle = False,
                  padding = False,
                  reverse = False,
                  dtype = torch.float16,
                  
                  # aug
                  ratio_aug = False,
                  ratio_range = [1., 0.2],
                  erode_aug = False,
                  erode_list = [1,2],
                  rotate_aug = False,
                  rotate_list = [0,90,180,270],
                  hflip_aug = False,
                  vflip_aug = False,
                ):

        self.annot = pd.read_csv(annot_path)
        self.img_dir = img_dir
        self.img_col = img_col
        self.vec_cols = vec_cols
        self.emb_dims = emb_dims
        self.dtype = dtype

        if bool(vec_cols):
            self.scaler = self.get_scaler()
        
        self.img_size = img_size 
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.reverse = reverse
        self.padding = padding
        
        self.ratio_aug  = ratio_aug
        self.ratio_range = ratio_range
        self.erode_aug = erode_aug
        self.erode_list = erode_list
        self.rotate_aug = rotate_aug
        self.rotate_list = rotate_list
        
        self.hflip_aug = hflip_aug
        self.vflip_aug = vflip_aug
        
    
    def img_pp(self, img_path):
        img = Image.open(img_path).convert('L')
        
        # black-white reverse. background should be black!
        if self.reverse:
            img = Image.eval(img, lambda x: 255 - x)
            
        # ratio aug
        if self.ratio_aug:
            while True:
                try:
                    m = self.ratio_range[0]
                    s = self.ratio_range[1]

                    img = ratio_img(img, np.random.normal(m,s))
                    break
                except:
                    pass
        
        # padding to image
        if self.padding: 
            img = pad_img(img, target_size=self.img_size)
        else:
            img = img.resize((self.img_size, self.img_size))

        # erode aug
        if self.erode_aug:
            ratio = int(np.random.choice(self.erode_list))
            if ratio != 1:
                img = erode_img(img, ratio)

        # rotate aug
        arr = np.array(img).astype('float')
        if self.rotate_aug:
            angle = int(np.random.choice(self.rotate_list))
            if angle != 0:
                arr = rotate_arr(arr, angle=angle)
                
        # to tensor 
        x = torch.tensor(arr, dtype=self.dtype).unsqueeze(0)

        # flip aug
        if self.hflip_aug:
            x = T.RandomHorizontalFlip()(x)
        if self.vflip_aug:
            x = T.RandomVerticalFlip()(x)

        return x

    def get_scaler(self, path=None):
        if path is None:
            target_params = self.annot[self.vec_cols]
            scaler = MinMaxScaler()
            results = scaler.fit_transform(target_params)
        else:
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
        
        return scaler
        
    def transfer_vec(self, vec):
        vf = [vec[0]] # volume fraction always first
        df = pd.DataFrame([vec[1:]], columns = self.vec_cols) 
        conds = self.scaler.transform(df)
        conds = list(conds[0])

        result = vf + conds
        result = torch.tensor(result, dtype=self.dtype)
        result = result.unsqueeze(-1).expand(-1,self.emb_dims)
        
        return result
        
    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        items = self.annot.iloc[idx]

        #img
        img_name = items[self.img_col]
        img_path = f'{self.img_dir}/{img_name}'
        img = self.img_pp(img_path)

        # basic volume fraction
        vec_conds = [get_vf_from_img(img)]
        
        # add vector conditions
        if bool(self.vec_cols):
            conds = list(items[self.vec_cols])
            vec_conds += conds

        vec_conds = self.transfer_vec(vec_conds)

        # norm
        img = img/ 127.5 - 1.

        return img, vec_conds

    
    def get_ds(self):
        return  DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_batch(self, idx=0):
        ds = self.get_ds()
        for i, batch in enumerate(ds):
            if i == idx : break
        return batch

        
def main():
    pass
    
if __name__ == "__main__":
    main()