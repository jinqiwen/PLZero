import os
import numpy as np
from PIL import Image
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

import glob
from torchvision.transforms.functional import crop

import torch.nn.functional as F

import ast
class NIHChestXray(Dataset):

    def __init__(self, args, pathDatasetFile, transform, classes_to_load='seen', exclude_all=True):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        self.num_classes = args.num_classes

        self._data_path = args.data_root
        self.args = args
        self.split_path = pathDatasetFile
        self.exclude_all = exclude_all
        
        self.classes_to_load = classes_to_load

        if self.args.dataset == 'CheXpert':
            
            self.CLASSES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']
        elif self.args.dataset == 'ChestX_Det10':
            self.CLASSES = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema', 'Fibrosis', 'Fracture', 'Mass', 'Nodule', 'Pneumothorax']

        else:
            self.df = pd.read_csv('data/CXR8/BBox_List_2017.csv')
            self.CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                        'Hernia']
            if args.plan == 0 :
                self.unseen_classes = ['Edema', 'Pneumonia', 'Emphysema', 'Fibrosis']
                self.seen_classes = ['Atelectasis', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                                      'Pneumothorax', 'Consolidation', 'Cardiomegaly', 'Pleural_Thickening', 'Hernia']
            elif args.plan == 1:
                self.unseen_classes = ['Pneumothorax', 'Effusion', 'Consolidation', 'Nodule']
                
                self.seen_classes = ['Cardiomegaly', 'Atelectasis', 'Edema', 'Mass', 'Emphysema',
                                     'Hernia', 'Pneumonia', 'Pleural_Thickening', 'Infiltration', 'Fibrosis']
            else:
                self.unseen_classes = ['Hernia', 'Pleural_Thickening', 'Atelectasis', 'Infiltration']
                
                self.seen_classes = ['Cardiomegaly', 'Effusion', 'Edema', 'Nodule', 'Emphysema',
                                     'Pneumothorax', 'Mass', 'Consolidation', 'Pneumonia', 'Fibrosis']

        self._class_ids = {v: i for i, v in enumerate(self.CLASSES) if v != 'No Finding'}

        if self.args.dataset == 'CXR8':
            self.seen_class_ids = [self._class_ids[label] for label in self.seen_classes]
            self.unseen_class_ids = [self._class_ids[label] for label in self.unseen_classes]
        self.classes_to_load = classes_to_load
        self.exclude_all = exclude_all
        if self.args.dataset == 'CXR8':
            self._construct_index()

    def _construct_index(self):
        
        max_labels = 0
        
        paths = glob.glob(f'{self._data_path}/**/images/*.png')
        
        self.names_to_path = {path.split('/')[-1]: path for path in paths}

        data_entry_file = 'Data_Entry_2017_v2020.csv'
        
        print(f'data partition path: {self.split_path}')
        with open(self.split_path, 'r') as f:
            file_names = f.readlines()

        split_file_names = np.array([file_name.strip().split(' ')[0].split('/')[-1] for file_name in file_names])
        df = pd.read_csv(f'{self._data_path}/{data_entry_file}')
        
        image_index = df.iloc[:, 0].values

        _, split_index, _ = np.intersect1d(image_index, split_file_names, return_indices=True)

        
        labels = df.iloc[:, 1].values
        
        labels = np.array(labels)[split_index]
        
        labels = [label.split('|') for label in labels]
        
        image_index = image_index[split_index]
        
        self._imdb = []
        self.class_ids_loaded = []
        
        im_path_hdf5=[]
        class_ids_hdf5=[]
        for index in range(len(split_index)):
            if len(labels[index]) == 1 and labels[index][0] == 'No Finding':
                 continue
            if self._should_load_image(labels[index]) is False:
                continue
            class_ids = [self._class_ids[label] for label in labels[index]]
            self.class_ids_loaded += class_ids
            self._imdb.append({
                'im_path': self.names_to_path[image_index[index]],
                'im_name': image_index[index],
                'labels': class_ids,
            })
            max_labels = max(max_labels, len(class_ids))
        
        self.class_ids_loaded = np.unique(np.array(self.class_ids_loaded))
        labels_matrix = torch.zeros((len(self._imdb)), self.args.num_classes)
        labels_ids = [x['labels']for x in self._imdb]
        for i in range(len(labels_ids)):
            for j in range(len(labels_ids[i])):
                labels_matrix[i][j] = 1
        label_co_matrix = self.label_co_matrix(labels_matrix, self.args.num_classes)
        self.label_co_occur_matrix = F.normalize(torch.from_numpy(label_co_matrix), p=2, dim=-1, eps=1e-12)
        print(f'Number of images: {len(self._imdb)}')
        print(f'Number of max labels per image: {max_labels}')
        print(f'Number of classes: {len(self.class_ids_loaded)}')
        print(f'unseen classes : {self.unseen_classes}')
    def label_co_matrix(self,labels_matrix,num_class):
        count_matrix = np.zeros((num_class, num_class))
        for i in range(labels_matrix.shape[0]):
            indices = np.where(labels_matrix[i] == 1)[0]
            for j in range(len(indices)):
                for k in range(j + 1, len(indices)):
                    count_matrix[indices[j], indices[k]] += 1
                    count_matrix[indices[k], indices[j]] += 1
        return count_matrix
    def _should_load_image(self, labels):
        selected_class_labels = self.CLASSES
        if self.classes_to_load == 'seen':
            selected_class_labels = self.seen_classes
        elif self.classes_to_load == 'unseen':
            selected_class_labels = self.unseen_classes
        elif self.classes_to_load == 'all':
            return True

        count = 0
        for label in labels:
            if label in selected_class_labels:
                count += 1

        if count == len(labels):
            
            return True
        elif count == 0:
            
            return False
        else:
            
            if self.exclude_all is True:
                return False
            else:
                return True
    def __getitem__(self, index):
        imagePath = self._imdb[index]['im_path']
        imageName = self._imdb[index]['im_name']
        imageData = Image.open(imagePath).convert('RGB')
        labels = torch.tensor(self._imdb[index]['labels'])
        labels = labels.unsqueeze(0)
        imageLabel = torch.zeros(labels.size(0), self.num_classes).scatter_(1, labels, 1.).squeeze()
        img = self.transform(imageData)
        return img, imageLabel
    
    def __len__(self):
        return len(self._imdb)

    def transform_add(self,imageName,imageData):

        imageLines = self.df.loc[self.df["Image Index"] == imageName]
        if isinstance(imageLines, pd.DataFrame):
            Label = imageLines['Finding Label']
            x = list(imageLines['Bbox [x'])
            y = list(imageLines['y'])
            w = list(imageLines['w'])
            h = list(imageLines['h]'])
            
            crop_image_data=[]
            for i in range(len(x)):
                crop_image_data.append(crop(imageData, x[i], y[i], h[i], w[i]))
            return crop_image_data

class CheXpert_Dataset_5(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:, 0])
        self.class_ids_loaded = [0,1,2,3,4]
        self.CLASSES = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Pleural Effusion']
        self.class_list = np.asarray(data_info.iloc[:,  [9, 3, 7, 6, 11]])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, [9, 3, 7, 6, 11]].head())
        self.transform = transform
    def __getitem__(self, index):
        
        img_path = os.path.join('data/CheXpert-v1.0-small/CheXpert-v1.0-small/CheXpert_test/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image,class_label
    def __len__(self):
        return len(self.img_path_list)
class CheXpert_Dataset_12_get_5(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 0])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 2:14])  
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index, :])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 2:14].head())
        self.transform = transform
        self.CLASSES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                        'Consolidation', 'Pneumonia',
                        'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']

        self.class_ids_loaded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.class_ids_loaded_test = [7, 1, 5, 4, 9]
    def __getitem__(self, index):
        img_path = os.path.join('data/CheXpert-v1.0-small/CheXpert-v1.0-small/CheXpert_test/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image, class_label
    def __len__(self):
        return len(self.img_path_list)

class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 0])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 1:15])
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index, :])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 1:15].head())
        self.transform = transform

        self.CLASSES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        self.class_ids_loaded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    def __getitem__(self, index):
        img_path = os.path.join('data/CheXpert-v1.0-small/CheXpert-v1.0-small/CheXpert_test/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image, class_label

    def __len__(self):
        return len(self.img_path_list)

class ChestX_Det10_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 0])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 3])
        self.class_list_all = [eval(item) for item in self.class_list_all]
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 3].head())
        self.transform = transform
        self.CLASSES = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema', 'Fibrosis',
                        'Fracture', 'Mass', 'Nodule', 'Pneumothorax']
        self.class_ids_loaded=[0,1,2,3,4,5,6,7,8,9]
    def __getitem__(self, index):
        img_path = os.path.join('data/ChestX-Det10/test_data/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image,class_label
    def __len__(self):
        return len(self.img_path_list)
class Covidx3_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 1])
        self.class_list_all = np.asarray(data_info.iloc[:, 2])
        self.class_list_all = [ ast.literal_eval(item) for item in self.class_list_all]
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 2].head())
        self.transform = transform
        self.CLASSES = ['normal','COVID-19']
        self.class_ids_loaded=[0,1]

    def __getitem__(self, index):
        img_path = os.path.join('data/COVIDx3/test/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image,class_label
    def __len__(self):
        return len(self.img_path_list)
class VinBigData_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 0])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 1])
        self.class_list_all = [ ast.literal_eval(item) for item in self.class_list_all]
        
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 1].head())
        self.transform = transform
        self.CLASSES = ['Aortic enlargement',
                 'Atelectasis',
                 'Pneumothorax',
                 'Lung Opacity',
                 'Pleural thickening',
                 'ILD',
                 'Pulmonary fibrosis',
                 'Calcification',
                 'Pleural effusion',
                 'Consolidation',
                 'Cardiomegaly',
                 'Other lesion',
                 'Nodule-Mass',
                 'Infiltration']
        self.class_ids_loaded = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    def __getitem__(self, index):
        img_path = os.path.join('data/VinBigData_Chest_X-ray/extract_test_png/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image,class_label
    def __len__(self):
        return len(self.img_path_list)

class ShenZhen_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 0])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 1])
        self.class_list_all = [ ast.literal_eval(item) for item in self.class_list_all]
        
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        print(data_info.iloc[:, 1].head())
        self.transform = transform
        self.CLASSES = ['normal','pulmonary tuberculosis']
        self.class_ids_loaded = [0,1]

    def __getitem__(self, index):
        img_path = os.path.join('data/Tuberculosis_Chest-X-rays_Shenzhen/images/images/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image,class_label
    def __len__(self):
        return len(self.img_path_list)

class ChestX_ray_14_Dataset(Dataset):
    def __init__(self, csv_path, transform):
        data_info = pd.read_csv(csv_path)
        self.img_path_list_all = np.asarray(data_info.iloc[:, 1])
        
        self.class_list_all = np.asarray(data_info.iloc[:, 2])
        self.class_list_all = [ ast.literal_eval(item) for item in self.class_list_all]
        
        self.img_path_list = []
        self.class_list = []
        for index in range(len(self.class_list_all)):
            self.class_list.append(self.class_list_all[index])
            self.img_path_list.append(self.img_path_list_all[index])
        self.class_list = np.array(self.class_list)
        self.img_path_list = np.array(self.img_path_list)
        
        self.transform = transform
        self.CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                        'Hernia']
        self.class_ids_loaded = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

        self.seen_class_ids = [0, 1, 2, 3, 4, 5, 7, 8, 12, 13]
        self.unseen_class_ids = [9, 6, 10, 11]
        print(f'ChestX_ray_14 test: {len(self.img_path_list)}')
    def __getitem__(self, index):
        img_path = os.path.join('data/CXR8/images/images/', self.img_path_list[index])
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)
        return image, class_label
    def __len__(self):
        return len(self.img_path_list)