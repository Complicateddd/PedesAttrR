import csv
import os
import os.path
# import tarfile
# from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
# import pickle
import torchvision.transforms as T


from sklearn.model_selection import train_test_split


class HehuangPAR(data.Dataset):
    def __init__(self,root = './hehuang/train2',phase = 'train', transform=None, target_transform=None,multi_scale = False):
       
        self.path_images = os.path.join(root,'train2_new')
        self.label_npy = np.load(os.path.join(root,'label.npy'),allow_pickle=True).item()
        

        # self.path_images = os.path.join(root,'release_data')
        # self.label_npy = np.load(os.path.join(root,'peta_100klabel.npy'),allow_pickle=True).item()

        self.transform = transform
        self.target_transform = target_transform
        
        

        self.attr_num = 46
        self.eval_attr_num =46

        self.image_name_list = []
        self.label_list = []
        for key_name in self.label_npy:
            self.image_name_list.append(key_name)
            self.label_list.append(self.label_npy[key_name])

        self.train_data,self.test_data,self.train_target,self.test_target =\
        train_test_split(self.image_name_list, self.label_list, test_size = 0.285,random_state=2022322)

        self.phase = phase
        self.multi_scale = multi_scale

        if phase=='train':
            self.image_name_list = self.train_data
            self.label_list = self.train_target
            print("=============>| Training set total {} samples |<==============".format(len(self.image_name_list)))
        else:
            self.image_name_list = self.test_data
            self.label_list = self.test_target
            print("=============>| Eval set total {} samples |<==============".format(len(self.image_name_list)))

    def __getitem__(self,index):
        path = self.image_name_list[index]
        target = self.label_list[index].astype(np.float32)

        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        
        # hh = img.size[1]
        # ww = img.size[0]
        # img = img.crop((0,int(hh*1/7.5),ww-1,int(hh*3.5/7.5)))


        sotmax_target_1 = target[:4].argmax()
        sotmax_target_2 = target[4:7].argmax()
        sotmax_target_3 = target[7:10].argmax()

        # sotmax_target = np.array(sotmax_target)
        # height = int(np.random.choice([256,280,224])) 
        # width = int(np.random.choice([128,192,100])) 

        height = 256
        width = 192 

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.phase == 'train':
            self.transform = T.Compose([
                T.Resize((height, width)),
                T.Pad(10),
                # T.RandomRotation(15),
                T.RandomCrop((height, width)),
                T.RandomHorizontalFlip(),
                # RandomErase(),
                T.ToTensor(),
                normalize,
            ])
        else:
            if self.multi_scale:

                self.transform1 = T.Compose([
                    T.Resize((256, 192)),
                    T.ToTensor(),
                    normalize
                ])

                self.transform2 = T.Compose([
                    T.Resize((280, 128)),
                    T.ToTensor(),
                    normalize
                ])

                self.transform3 = T.Compose([
                    T.Resize((224, 192)),
                    T.ToTensor(),
                    normalize
                ])

                self.transform4 = T.Compose([
                    T.Resize((256, 100)),
                    T.ToTensor(),
                    normalize
                ])

                self.transform5 = T.Compose([
                    T.Resize((280, 192)),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((256, 192)),
                    T.ToTensor(),
                    normalize
                ])

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        

        if self.phase == 'train':
            img = self.transform(img)
            return img,target,path,sotmax_target_1,sotmax_target_2,sotmax_target_3,0,0,0,0

        else:
            if self.multi_scale:
                img_1 = self.transform1(img)
                img_2 = self.transform2(img)
                img_3 = self.transform3(img)
                img_4 = self.transform4(img)
                img_5 = self.transform4(img)

                return img_1,target,path,sotmax_target_1,sotmax_target_2,sotmax_target_3,img_2,img_3,img_4,img_5
            else:
                img = self.transform(img)
                return img,target,path,sotmax_target_1,sotmax_target_2,sotmax_target_3,0,0,0,0




    def __len__(self):
        return len(self.image_name_list)


class HehuangPARTest(data.Dataset):
    def __init__(self,root = './hehuang',phase = 'test', transform=None, target_transform=None):
        
        self.path_images = os.path.join(root,'test2A_new')
        self.transform = transform
        self.target_transform = target_transform
        # print(self.path_images)

        self.image_name_list = os.listdir(self.path_images)
        # print(self.image_name_list[0])

        self.image_name_list = sorted(self.image_name_list,key=lambda x: int(x[-15:-4]))

        # print(self.image_name_list)
        self.attr_num = 46
        self.eval_attr_num =46

        # self.image_name_list = []
        # self.label_list = []

        print("=============>| Test set total {} samples |<==============".format(len(self.image_name_list)))

    def __getitem__(self,index):
        
        path = self.image_name_list[index]
        # target = self.label_list[index].astype(np.float32)

        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img,0,path

    def __len__(self):
        return len(self.image_name_list)





class HehuangVAR(data.Dataset):
    def __init__(self,root = './hehuang/car',phase = 'train', transform=None, target_transform=None):
        self.path_images = os.path.join(root,'train')
        self.transform = transform
        self.target_transform = target_transform
        
        self.label_npy = np.load(os.path.join(root,'label.npy'),allow_pickle=True).item()

        self.attr_num = 4
        self.eval_attr_num =4

        self.image_name_list = []
        self.label_list = []
        for key_name in self.label_npy:
            self.image_name_list.append(key_name)
            self.label_list.append(self.label_npy[key_name])

        self.train_data,self.test_data,self.train_target,self.test_target =\
        train_test_split(self.image_name_list, self.label_list, test_size = 0.25,random_state=2022)

        if phase=='train':
            self.image_name_list = self.train_data
            self.label_list = self.train_target
            print("=============>| Training set total {} samples |<==============".format(len(self.image_name_list)))
        else:
            self.image_name_list = self.test_data
            self.label_list = self.test_target
            print("=============>| Eval set total {} samples |<==============".format(len(self.image_name_list)))

    def __getitem__(self,index):
        path = self.image_name_list[index]
        target = self.label_list[index].astype(np.float32)

        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target,path

    def __len__(self):
        return len(self.image_name_list)


class HehuangVARTest(data.Dataset):
    def __init__(self,root = './hehuang/car',phase = 'test', transform=None, target_transform=None):
        
        self.path_images = os.path.join(root,'testA')
        self.transform = transform
        self.target_transform = target_transform
        # print(self.path_images)
        
        self.image_name_list = os.listdir(self.path_images)
        # print(self.image_name_list)

        self.image_name_list = sorted(self.image_name_list,key=lambda x: int(x.split('_')[1].replace('.jpg','')))

        self.attr_num = 4
        self.eval_attr_num =4

        # self.image_name_list = []
        # self.label_list = []

        print("=============>| Test set total {} samples |<==============".format(len(self.image_name_list)))

    def __getitem__(self,index):
        
        path = self.image_name_list[index]
        # target = self.label_list[index].astype(np.float32)

        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img,0,path

    def __len__(self):
        return len(self.image_name_list)

import torch.nn.functional as F

def collate_fn(batch):
    img_list = [x[0] for x in batch]

    max_height = max([x[0].shape[1] for x in batch])
    max_width = max([x[0].shape[2] for x in batch])

    # print(max_height)
    # print(max_width)


    target = [torch.from_numpy(x[1]).unsqueeze(0) for x in batch]
    path_list = [x[2] for x in batch]
    sotmax_target_1 = [torch.tensor(x[3]).unsqueeze(0) for x in batch]
    sotmax_target_2 = [torch.tensor(x[4]).unsqueeze(0) for x in batch]
    sotmax_target_3 = [torch.tensor(x[5]).unsqueeze(0) for x in batch]

    for i in range(len(img_list)):

        # print(img_list[i].shape)

        # max_height
        pad_width_left = int(max_width-img_list[i].shape[2])//2
        pad_width_right = int(max_width-img_list[i].shape[2]) - pad_width_left

        pad_height_left = int(max_height-img_list[i].shape[1])//2
        pad_height_right = int(max_height-img_list[i].shape[1]) - pad_height_left

        img_list[i] = F.pad(img_list[i], (pad_width_left,pad_width_right,pad_height_left,pad_height_right)).unsqueeze(0)

        # print(target[i].shape)
    # print(target)

    img = torch.cat(img_list,0)
    target = torch.cat(target,0)
    sotmax_target_1 = torch.cat(sotmax_target_1,0)
    sotmax_target_2 = torch.cat(sotmax_target_2,0)
    sotmax_target_3 = torch.cat(sotmax_target_3,0)
    # print(img.shape)



    return img,target,path_list,sotmax_target_1,sotmax_target_2,sotmax_target_3,0,0,0,0