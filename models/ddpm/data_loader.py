import glob, os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from data_utils import *


class ImageLoader(Dataset):
    def __init__( self, 
                  data_dir,
                  img_fomat='jpg',
                  img_size=128,
                  batch_size=8,
                 
                  shuffle = True,
                  padding = False,
                  reverse = False,
                 
                  ratio_aug = False,
                  ratio_range = 0.2,
                  erode_aug = False,
                  erode_list = [1,2],
                  rotate_aug = False,
                  rotate_list = [0,90,180,270],
                 
                  hflip_aug = False,
                  vflip_aug = False,
                ):

        self.img_fomat = img_fomat
        self.x_paths = sorted(glob.glob(f'{data_dir}/*{self.img_fomat}'))
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

    
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        img = Image.open(self.x_paths[idx]).convert('L')
        
        # black-white reverse. background should be black!
        if self.reverse:
            img = Image.eval(img, lambda x: 255 - x)

        # ratio aug
        if self.ratio_aug:
            while True:
                try:
                    img = ratio_img(img, np.random.normal(1., self.ratio_range))
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
                arr = roatate_arr(arr, angle=angle)
                
        # to tensor 
        x = torch.tensor(arr, dtype=torch.float16).unsqueeze(0)

        # flip aug
        if self.hflip_aug:
            x = T.RandomHorizontalFlip()(x)
        if self.vflip_aug:
            x = T.RandomVerticalFlip()(x)
            
        # norm
        x = x / 127.5 - 1.

        return x

    
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