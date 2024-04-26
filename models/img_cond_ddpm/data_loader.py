import glob, os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from utils import *

class Loader(Dataset):
    def __init__( self, 
                  data_dir,
                  img_fomat='jpg',
                  img_cond_dirs = [],
                 
                  img_size=128,
                  batch_size=8,
                 
                  shuffle = True,
                  padding = False,
                  reverse = False,

                  ratio_aug = False,
                  ratio_range = [1., 0.2],
                  rotate_aug = False,
                  rotate_list = [0,90,180,270],
                 
                  hflip_aug = False,
                  vflip_aug = False,
                ):

        self.img_fomat = img_fomat
        self.x_paths = sorted(glob.glob(f'{data_dir}/*{self.img_fomat}'))
        self.img_cond_dirs = img_cond_dirs
        
        self.img_size = img_size 
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.reverse = reverse
        self.padding = padding
        
        self.rotate_aug = rotate_aug
        self.rotate_list = rotate_list
        
        self.hflip_aug = hflip_aug
        self.vflip_aug = vflip_aug

    
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert('L')

        img_conds = []
        for img_cond_dir in self.img_cond_dirs:
            img_name = self.x_paths[idx].split('/')[-1]
            img_cond = Image.open(f'{img_cond_dir}/{img_name}').convert('L')
            img_conds.append(img_cond)
        
        # black-white reverse. background should be black!
        if self.reverse:
            img = Image.eval(img, lambda x: 255 - x)
        
        # padding to image
        if self.padding: 
            img = pad_img(img, target_size=self.img_size)
            
            for i, img_cond in enumerate(img_conds):
                img_conds[i] = pad_img(img_cond, target_size=self.img_size)
                
        else:
            img = img.resize((self.img_size, self.img_size))
            for i, img_cond in enumerate(img_conds):
                img_conds[i] = img_cond.resize((self.img_size, self.img_size))
                
        
        # rotate aug
        arr = np.array(img).astype('float')
        arr_conds = [np.array(img) for img in img_conds]

        if self.rotate_aug:
            angle = int(np.random.choice(self.rotate_list))
            if angle != 0:
                arr = rotate_arr(arr, angle=angle)
                for i, arr_cond in enumerate(arr_conds):
                    arr_conds[i] = rotate_arr(arr_cond, angle=angle)
        
        arr_conds = np.stack(arr_conds)
        
        # to tensor 
        x = torch.tensor(arr, dtype=torch.float16).unsqueeze(0)
        img_conds = torch.tensor(arr_conds, dtype=torch.float16)
        
        # flip aug
        if self.hflip_aug:
            hflip = T.RandomHorizontalFlip(p=1.0)
            x = hflip(x)
            img_conds = hflip(img_conds)
        if self.vflip_aug:
            vflip = T.RandomVerticalFlip(p=1.0)
            x = vflip(x)
            img_conds = vflip(img_conds)

        # volume fraction
        vf = get_vf_from_img(x)
        vf = torch.tensor(vf, dtype=torch.float16)
        
        # norm
        x = x / 127.5 - 1.
        img_conds = img_conds / 127.5 - 1.
        
        return x, vf, img_conds

    def get_loader(self):
        return  DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_batch(self, idx=0):
        loader = self.get_loader()
        for i, batch in enumerate(loader):
            if i == idx : break
        return batch
        
def main():
    pass
    
if __name__ == "__main__":
    main()