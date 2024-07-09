import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from sklearn.metrics import accuracy_score
from models.zsl_models import ZSLNet
from dataset import NIHChestXray,CheXpert_Dataset_5, CheXpert_Dataset, ChestX_Det10_Dataset, Covidx3_Dataset, VinBigData_Dataset,ShenZhen_Dataset, ChestX_ray_14_Dataset,CheXpert_Dataset_12_get_5

from plots import plot_array

from torch import nn
from sklearn.metrics import average_precision_score, matthews_corrcoef
from datetime import datetime

from sklearn import manifold

from matplotlib import pyplot as plt
import seaborn as sns


class ChexnetTrainer(object):
    def __init__(self, args):
        self.args = args

        self.device = torch.device(f'cuda:{self.args.cuda_num}' if torch.cuda.is_available() else 'cpu')

        self.textual_embeddings = np.load(args.textual_embeddings, allow_pickle=True)

        self.model = ZSLNet(self.args, self.textual_embeddings, self.device).to(self.device)




        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=1e-5)


        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')

        self.loss = torch.nn.BCELoss(size_average=True)

        self.auroc_min_loss = 0.0

        self.start_epoch = 1

        self.lossMIN = float('inf')

        self.mAPMAX = float('-inf')

        self.f1Max = float('-inf')


        self.max_auroc_mean = float('-inf')

        self.best_epoch = 1

        self.train_losses = []

        self.val_losses = []

        print(f'\n\nloaded imagenet weights {self.args.pretrained}\n\n\n')

        self.resume_from()

        self.load_from()

        if self.args.dataset == 'CheXpert_5':
            self.init_cheXpert_test_dataset_5()
        elif self.args.dataset == 'CheXpert':
            self.init_cheXpert_test_dataset()
        elif self.args.dataset == 'CheXpert_12_5':
            self.init_cheXpert_test_dataset_12_5()
        elif self.args.dataset == 'chestXDet_10':
            self.init_chestX_Det_10_test_dataset()
        elif self.args.dataset == 'COVIDx3':
            self.init_covidx_test_dataset()
        elif self.args.dataset == 'VinBigData':
            self.init_vig_big_data_test_dataset()
        elif self.args.dataset == 'ShenZhen':
            self.init_shenzhen_test_dataset()
        elif self.args.dataset == 'ChestX-ray14':
            self.init_chestX_ray_14_test_dataset()
        else:
            self.init_dataset()

        self.steps = [int(step) for step in self.args.steps.split(',')]

        self.time_start = time.time()

        self.time_end = time.time()

        self.should_test = False

        if self.args.dataset == 'CXR8':
            self.model.class_ids_loaded = self.train_dl.dataset.class_ids_loaded

        self.show_model=True
        self.k1 = 2
        self.k1 = 3
        self.threshold = 0.5
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def __call__(self):
        self.train()

    def step_lr(self, epoch):
        step = self.steps[0]
        for index, s in enumerate(self.steps):
            if epoch < s:
                break
            else:
                step = s

        lr = self.args.lr * (0.1 ** (epoch // step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_from(self):
        if self.args.load_from is not None:
            checkpoint = torch.load(self.args.load_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded checkpoint from {self.args.load_from}')

    def resume_from(self):
        if self.args.resume_from is not None:
            checkpoint = torch.load(self.args.resume_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.lossMIN = checkpoint['lossMIN']
            self.max_auroc_mean = checkpoint['max_auroc_mean']
            print(f'resuming training from epoch {self.start_epoch}')

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'max_auroc_mean': self.max_auroc_mean,
                'optimizer': self.optimizer.state_dict(),
                'lossMIN': self.lossMIN
            }, path)
        print(f"saving {prefix} checkpoint")

    def init_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        train_transforms = []
        train_transforms.append(transforms.RandomResizedCrop(self.args.crop))

        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)

        datasetTrain = NIHChestXray(self.args, self.args.train_file, transform=transforms.Compose(train_transforms))

        self.train_dl = DataLoader(dataset=datasetTrain, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                                   pin_memory=True)
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

        datasetVal = NIHChestXray(self.args, self.args.val_file, transform=transforms.Compose(test_transforms))
        self.val_dl = DataLoader(dataset=datasetVal, batch_size=int(self.args.batch_size*2), shuffle=False,
                                 num_workers=4, pin_memory=True)
        datasetTest = ChestX_ray_14_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        datasetTest_cp = CheXpert_Dataset_5(self.args.test_file_cp, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 2), num_workers=8,
                                  shuffle=False, pin_memory=True)
        self.test_dl_cp = DataLoader(dataset=datasetTest_cp, batch_size=int(self.args.batch_size * 2), num_workers=8,
                                  shuffle=False, pin_memory=True)

        print(datasetTest.CLASSES)
    def init_cheXpert_test_dataset_5(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        datasetTest = CheXpert_Dataset_5(self.args.test_file, transform=transforms.Compose(test_transforms))
        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)

    def init_cheXpert_test_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        datasetTest = CheXpert_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)
    def init_cheXpert_test_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        datasetTest = CheXpert_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)
    def init_cheXpert_test_dataset_12_5(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        datasetTest = CheXpert_Dataset_12_get_5(self.args.test_file, transform=transforms.Compose(test_transforms))
        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)
    def init_chestX_Det_10_test_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

        datasetTest = ChestX_Det10_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)

    def init_covidx_test_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

        datasetTest = Covidx3_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)

    def init_vig_big_data_test_dataset(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

        datasetTest = VinBigData_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)

        print(datasetTest.CLASSES)

    def init_shenzhen_test_dataset(self):

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

        datasetTest = ShenZhen_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))

        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 3), num_workers=8,
                                  shuffle=False, pin_memory=True)

        print(datasetTest.CLASSES)
    def init_chestX_ray_14_test_dataset(self):
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))


        datasetTest = ChestX_ray_14_Dataset(self.args.test_file, transform=transforms.Compose(test_transforms))
        self.test_dl = DataLoader(dataset=datasetTest, batch_size=int(self.args.batch_size * 6), num_workers=8,
                                  shuffle=False, pin_memory=True)

        print(datasetTest.CLASSES)
    def train(self):

        for self.epoch in range(self.start_epoch, self.args.epochs):

            epoch_idx = self.epoch

            self.epochTrain()

            lossVal, val_ind_auroc, val_ind_f1, val_ind_acc, val_ind_mcc, mAP= self.epochVal()


            val_ind_auroc = np.array(val_ind_auroc)

            aurocMean = val_ind_auroc.mean()

            self.save_checkpoint(prefix=f'last_epoch')

            self.should_test = False

            if aurocMean > self.max_auroc_mean:

                self.max_auroc_mean = aurocMean

                self.save_checkpoint(prefix='best_auroc')

                self.best_epoch = self.epoch

                self.should_test = True
            if lossVal < self.lossMIN:

                self.lossMIN = lossVal

                self.auroc_min_loss = aurocMean

                self.save_checkpoint(prefix='min_loss')

                self.should_test = True

            self.print_auroc(val_ind_auroc, self.val_dl.dataset.class_ids_loaded, prefix='val')
            self.print_base_indicator(val_ind_mcc, 'mcc', prefix='val')
            self.print_base_indicator(val_ind_acc, 'acc', prefix='val')
            self.print_base_indicator(val_ind_f1, 'f1_score', prefix='val')
            self.print_base_indicator(mAP, 'mAP', prefix='val')

            if self.should_test is True:
                test_ind_auroc, test_ind_f1, test_ind_acc, test_ind_mcc, mAP=self.test()
                test_ind_auroc = np.array(test_ind_auroc)
                self.write_results(val_ind_auroc, self.val_dl.dataset.class_ids_loaded,prefix=f'\n\nepoch {self.epoch}\nval', mode='a')
                self.write_results(test_ind_auroc[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen', mode='a')
                self.write_results(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen', mode='a')
                self.print_auroc(test_ind_auroc[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
                self.print_auroc(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

                self.print_base_indicator(test_ind_mcc, 'mcc', prefix='\ntest')
                self.print_base_indicator(test_ind_acc, 'acc',  prefix='\ntest')
                self.print_base_indicator(test_ind_f1, 'f1_score', prefix='\ntest')
                self.print_base_indicator(mAP, 'mAP', prefix='\ntest')
                self.write_base_indicator_results(mAP, self.val_dl.dataset.class_ids_loaded, prefix='\ntest_test_mAP', mode='a')

                test_ind_auroc_cp, f1_mean_cp, acc_mean_cp, mcc_mean_cp, mAP_cp = self.test_cheXpert_5_when_train()
                test_ind_auroc_cp = np.array(test_ind_auroc_cp)
                self.print_auroc_cheXpert(test_ind_auroc_cp, [0, 1, 2, 3, 4], prefix='test')
                self.print_base_indicator(mcc_mean_cp, 'Mcc', prefix='cp_test')
                self.print_base_indicator(f1_mean_cp, 'F1_score', prefix='cp_test')
                self.print_base_indicator(acc_mean_cp, 'Acc', prefix='cp_test')
                self.print_base_indicator(mAP_cp, 'mAP', prefix='cp_test')

            plot_array(self.val_losses, f'{self.args.save_dir}/val_loss')
            
            print(f'best epoch {self.best_epoch} best auroc {self.max_auroc_mean} loss {lossVal:.6f} auroc at min loss {self.auroc_min_loss:0.4f}')
            
            self.scheduler.step(lossVal)

    def get_eta(self, epoch, iter):
        self.time_end = time.time()
        delta = self.time_end - self.time_start
        delta = delta * (len(self.train_dl) * (self.args.epochs - epoch) - iter)
        sec = timedelta(seconds=int(delta))
        d = (datetime(1, 1, 1) + sec)
        eta = f"{d.day - 1} Days {d.hour}:{d.minute}:{d.second}"
        self.time_start = time.time()
        return eta

    def epochTrain(self):

        self.model.train()

        epoch_loss = 0

        for batchID, (inputs, target) in enumerate(self.train_dl):

            target = target.to(self.device)

            inputs = inputs.to(self.device)

            output, loss = self.model(inputs, target)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            eta = self.get_eta(self.epoch, batchID)

            epoch_loss += loss.item()

            if batchID % 10 == 9:

                print(
                    f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] eta: {eta:<20} [{batchID:04}/{len(self.train_dl)}] lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss / batchID:0.5f}")

        self.train_losses.append(round(epoch_loss / batchID,3))
        print(f'epoch {self.epoch} train_loss:{epoch_loss / batchID}')
        plot_array(self.train_losses, f'{self.args.save_dir}/train_loss')

    def epochVal(self):

        self.model.eval()

        lossVal = 0

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        for i, (inputs, target) in enumerate(tqdm(self.val_dl)):

            with torch.no_grad():

                target = target.to(self.device)

                inputs = inputs.to(self.device)

                varTarget = torch.autograd.Variable(target)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                varOutput, losstensor = self.model(varInput, varTarget, n_crops=n_crops, bs=bs)

                varOutput = self.softmax(varOutput)
                outPRED = torch.cat((outPRED, varOutput), 0)

                outGT = torch.cat((outGT, target), 0)

                lossVal += losstensor.item()

                del varOutput, varTarget, varInput, target, inputs

        lossVal = lossVal / len(self.val_dl)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.val_dl.dataset.class_ids_loaded)

        n_class = len(self.val_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        self.val_losses.append(lossVal)

        threshold = [0.5] * n_class
        Mccs = self.compute_mccs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold,
                                           n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold,
                                         n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold,
                                           n_class)
        print(f'lossVal:{lossVal}')
        print(f'Mccs:{Mccs}')
        print(f'F1s:{F1s}')
        print(f'Accs:{Accs}')

        return lossVal, aurocIndividual,  F1s[-1], Accs[-1], Mccs[-1], mAP


    def test(self):

        cudnn.benchmark = True

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()

        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops=n_crops, bs=bs)
                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        threshold = [0.5] * n_class


        Mccs = self.compute_mccs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold, n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold, n_class)

        print(f'Mccs:{Mccs}')
        print(f'F1s:{F1s}')
        print(f'Accs:{Accs}')

        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_tsne(self):

        cudnn.benchmark = True


        outGT = torch.FloatTensor().to(self.device)


        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()

        x_tsne = []
        x_tsne_gener = []
        y_tsne = []
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.to(self.device)

                outGT = torch.cat((outGT, target), 0)


                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _, visual_feats, visual_feats_fantasy = self.model(varInput, n_crops=n_crops, bs=bs)

                x_tsne.append(visual_feats)
                x_tsne_gener.append(visual_feats_fantasy)
                y_tsne.append(target)

                outPRED = torch.cat((outPRED, out.data), 0)

        input_tsne, label_tsne = self.generate_tsne_input_and_label(x_tsne, y_tsne)

        input_tsne_gener = self.generate_tsne_input(x_tsne_gener)
        self.train_tsne_and_plot_2(input_tsne, label_tsne, input_tsne_gener)

    def test_cheXpert_5_when_train(self):
        cudnn.benchmark = True

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)
        class_index = [0, 1, 8, 9, 2]

        self.model.eval()
        for i, (inputs, target) in enumerate(tqdm(self.test_dl_cp)):

            with torch.no_grad():

                target = target.float().to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, class_index=class_index, n_crops=n_crops, bs=bs)

                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl_cp.dataset.class_ids_loaded)

        n_class = len(self.test_dl_cp.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy(),5)

        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)

        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP

    def test_cheXpert_5(self):

        cudnn.benchmark = True


        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.float().to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops=n_crops, bs=bs)

                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)


        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy(),5)
        
        
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_cheXpert(self):
        cudnn.benchmark = True

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.float().to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops=n_crops, bs=bs)

                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy(), n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP

    def test_cheXpert_12_get_5(self):
        cudnn.benchmark = True

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.float().to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops=n_crops, bs=bs)

                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)
        class_ids_loaded = self.test_dl.dataset.class_ids_loaded_test

        aurocIndividual = self.computeAUROC(outGT, outPRED, class_ids_loaded)

        n_class = len(class_ids_loaded)
        outGT = outGT[:,class_ids_loaded]
        outPRED = outPRED[:,class_ids_loaded]
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy(),n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(), threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_chestX_Det_10(self):

        cudnn.benchmark = True

        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)

        self.model.eval()

        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):

            with torch.no_grad():

                target = target.float().to(self.device)

                outGT = torch.cat((outGT, target), 0)

                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops = n_crops, bs=bs)

                out = torch.softmax(out, dim=-1)


                outPRED = torch.cat((outPRED, out.data), 0)


        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy(),10)
        PointGames = self.compute_pointing_game(outGT.cpu().numpy(), outPRED.cpu().numpy(), threshold)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(PointGames)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_covidx(self):
        
        cudnn.benchmark = True
        
        outGT = torch.FloatTensor().to(self.device)

        outPRED = torch.FloatTensor().to(self.device)
        
        self.model.eval()  

        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):
            
            with torch.no_grad():
                
                target = target.float().to(self.device)
                
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = inputs.size()
                
                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops = n_crops, bs=bs)
                
                
                out = torch.sigmoid(out)
                
                outPRED = torch.cat((outPRED, out.data), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())
        
        outPRED = self.sigmoid(outPRED)

        threshold = [0.5] * n_class
        Mccs = self.compute_mccs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_vinbigdata(self):
        
        cudnn.benchmark = True
        
        
        outGT = torch.FloatTensor().to(self.device)
        
        
        outPRED = torch.FloatTensor().to(self.device)
        
        self.model.eval()  

        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):
            
            with torch.no_grad():
                
                target = target.float().to(self.device)
                
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = inputs.size()
                
                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                out, _ = self.model(varInput, n_crops = n_crops, bs=bs)
                
                out = torch.softmax(out, dim=-1)

                outPRED = torch.cat((outPRED, out.data), 0)

        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)

        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())

        Mccs, threshold = self.compute_mccs(outGT.cpu().numpy(), outPRED.cpu().numpy())
        
        
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def test_shenzhen(self):
        
        cudnn.benchmark = True
        
        
        outGT = torch.FloatTensor().to(self.device)
        
        
        outPRED = torch.FloatTensor().to(self.device)
        
        self.model.eval()  

        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):
            
            with torch.no_grad():
                
                target = target.float().to(self.device)
                
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = inputs.size()
                
                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))
                
                
                
                out, _ = self.model(varInput, n_crops = n_crops, bs=bs)
                
                
                
                
                outPRED = torch.cat((outPRED, out.data), 0)

        
        aurocIndividual = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)
        
        
        
        
        
        
        
        
        
        
        n_class = len(self.test_dl.dataset.class_ids_loaded)
        mAP = self.mAP(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy())
        
        outPRED = self.sigmoid(outPRED)
        
        
        
        
        
        
        
        
        threshold = [0.5] * n_class
        Mccs = self.compute_mccs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        F1s = self.compute_F1s_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                         threshold, n_class)
        Accs = self.compute_Accs_threshold(outGT.cpu().detach().numpy(), outPRED.cpu().detach().numpy(),
                                           threshold, n_class)
        print(f'{Mccs}')
        print(f'{F1s}')
        print(f'{Accs}')
        return aurocIndividual, F1s[-1], Accs[-1], Mccs[-1], mAP
    def train_tsne_and_plot(self,input_tsne,label_tsne):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=500)
        X_tsne = tsne.fit_transform(input_tsne)
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        
        palette = sns.color_palette('tab20b_r', n_colors=14)
        

        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=label_tsne, legend='full', palette=palette)
        
        plt.legend(loc='upper right')
        
        plt.savefig('scatterplot.png', dpi=300)  
    def train_tsne_and_plot_2(self,input_tsne,label_tsne,input_tsne_gener):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, n_iter=250)
        X = np.concatenate((input_tsne, input_tsne_gener), axis=0)
        X_tsne = tsne.fit_transform(X)
        sns.set_style("white")
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        X_label_tsne = X_tsne[:len(input_tsne)]
        X_unlabel_tsne = X_tsne[len(input_tsne):]
        palette = ['#f47c7c' if label in [6, 9, 10, 11] else '#a1de93' for label in label_tsne]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.scatterplot(x=X_label_tsne[:, 0], y=X_label_tsne[:, 1], hue=label_tsne, palette=palette, sizes=[0.2], ax=ax[0])
        sns.scatterplot(x=X_unlabel_tsne[:, 0], y=X_unlabel_tsne[:, 1], color='#70a1d7', sizes=[0.2], ax=ax[1])
        ax[0].set_title('Labeled Data')
        ax[1].set_title('Unlabeled Data')
        plt.show()
        
    def generate_tsne_input_and_label(self, x_tsne, y_tsne):
        t_features = []
        t_labels = []
        for i in range(len(x_tsne)):
            for j in range(x_tsne[i].shape[0]):
                label_index = torch.where(y_tsne[i][j] > 0)[0]
                for lab in label_index:
                    t_features.append(x_tsne[i][j].unsqueeze(0))
                    t_labels.append(lab)
        input_tsne = torch.cat(t_features, dim=0).cpu().numpy()
        label_tsne = np.zeros(len(t_labels))
        for i in range(len(t_labels)):
            label_tsne[i] = t_labels[i].cpu().numpy()
        return input_tsne, label_tsne
    def generate_tsne_input(self, x_tsne):
        t_features = []
        for i in range(len(x_tsne)):
            for j in range(x_tsne[i].shape[0]):
                    t_features.append(x_tsne[i][j].unsqueeze(0))
        input_tsne = torch.cat(t_features, dim=0).cpu().numpy()
        return input_tsne
    def computeAUROC(self, dataGT, dataPRED, class_ids):
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        for i in class_ids:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        return outAUROC

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    def binarize_predictions(self, predictions, thresholds):
        binarized_predictions = []

        for pred in predictions:
            binarized_pred = [1 if p >= t else 0 for p, t in zip(pred, thresholds)]
            binarized_predictions.append(binarized_pred)

        return binarized_predictions

    def compute_pointing_game(self, reference, predictions, thresholds):
        predictions = self.binarize_predictions(predictions, thresholds)
        num_samples = len(reference)
        num_correct = 0

        for i in range(num_samples):
            ref_labels = reference[i]
            pred_labels = predictions[i]

            if set(ref_labels) == set(pred_labels):
                num_correct += 1

        pointing_game_score = num_correct / num_samples
        return pointing_game_score
    def compute_F1(self, predictions, labels, mode_F1, k_val):
        
        idx = predictions.topk(dim=1, k=k_val)[1]
        predictions.fill_(0)
        predictions.scatter_(dim=1, index=idx, src=torch.ones(predictions.size(0), k_val).to(self.device))
        if mode_F1 == 'overall':
            
            mask = predictions == 1
            TP = (labels[mask] == 1).sum().float()
            tpfp = mask.sum().float()
            tpfn = (labels == 1).sum().float()
            p = TP / tpfp
            r = TP / tpfn
            f1 = 2 * p * r / (p + r)
        else:
            num_class = predictions.shape[1]
            
            f1 = np.zeros(num_class)
            p = np.zeros(num_class)
            r = np.zeros(num_class)
            for idx_cls in range(num_class):
                prediction = np.squeeze(predictions[:, idx_cls])
                label = np.squeeze(labels[:, idx_cls])
                if np.sum(label > 0) == 0:
                    continue
                binary_label = np.clip(label, 0, 1)
                f1[idx_cls] = f1_score(binary_label, prediction)
                p[idx_cls] = precision_score(binary_label, prediction)
                r[idx_cls] = recall_score(binary_label, prediction)
        return  p.cpu().numpy(), r.cpu().numpy(), f1.cpu().numpy()

    def average_precision(self, output, target):
        epsilon = 1e-8

        
        indices = output.argsort()[::-1]
        total_count_ = np.cumsum(np.ones((len(output), 1)))
        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)
        return precision_at_i

    def mAP(self, targs, preds):
        """Returns the model's mean of  average precision for each class
        """

        if np.size(preds) == 0:
            return 0
        ap = np.zeros((preds.shape[1]))
        
        for k in range(preds.shape[1]):
            
            scores = preds[:, k]
            targets = targs[:, k]
            
            ap[k] = self.average_precision(scores, targets)
        print(ap)
        return 100 * ap.mean()

    

    def write_results(self, aurocIndividual, class_ids, prefix='val', mode='a'):
        

        with open(f"{self.args.save_dir}/results.txt", mode) as results_file:
            
            
            now = datetime.now()
            aurocMean = aurocIndividual.mean()
            
            results_file.write(f'{now}--{prefix} AUROC mean {aurocMean:0.4f}\n')
            for i, class_id in enumerate(class_ids):
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}\n')
    def write_pred_and_gt(self, outPred, outGT, prefix='val', mode='a'):
        
        with open(f"{self.args.save_dir}/results_pred_gt.txt", mode) as results_file:
            
            
            
            
            
            
            for i, igt in enumerate(outGT):
                results_file.write(f'{prefix}_label {igt} \n{prefix}_pred {outPred[i]}')
    def compute_F1_2022(self, predictions, labels):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        predictions[predictions>0] = 1
        predictions[predictions<=0] = 0
        mask = predictions == 1
        
        
        
        
        
        result = self.count_different_elements(predictions,labels)
        print(f'预测错误类别统计：{result}')
        TP = (labels[mask] == 1).sum()
        print(f'TP：{TP}')
        tpfp = mask.sum()
        tpfn = (labels == 1).sum()
        p = TP / tpfp
        r = TP / tpfn
        f1 = 2 * p * r / (p + r)
        return p, r , f1

    def count_different_elements(self,matrix1, matrix2):
        
        equal_elements = torch.eq(matrix1, matrix2)

        
        different_elements = torch.sum(~equal_elements, dim=0)

        return different_elements
    def write_base_indicator_results(self, baseIndicator, class_ids, prefix='val', mode='a'):
        
        with open(f"{self.args.save_dir}/results.txt", mode) as results_file:
            
            
            
            results_file.write(f'{prefix} {baseIndicator:0.4f}\n')
            
            

    def print_auroc(self, aurocIndividual, class_ids, prefix='val'):
        
        aurocMean = aurocIndividual.mean()
        
        print(f'{prefix} AUROC mean {aurocMean:0.4f}')
        
        for i, class_id in enumerate(class_ids):
            
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}')
    def print_auroc_test(self, aurocIndividual, class_ids, prefix='val'):
        
        aurocMean = aurocIndividual.mean()
        
        print(f'{prefix} AUROC mean {aurocMean:0.4f}')
        
        for i, class_id in enumerate(class_ids):
            
            print(f'{self.test_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}')
    def print_auroc_cheXpert(self, aurocIndividual, class_ids, prefix='test'):
        
        aurocMean = aurocIndividual.mean()
        
        print(f'{prefix} AUROC mean {aurocMean:0.4f}')
        
        for i, class_id in enumerate(class_ids):
            
            print(f'{self.test_dl_cp.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}')
    def print_auroc_cheXpert_for_test(self, aurocIndividual, class_ids, prefix='test'):
        
        aurocMean = aurocIndividual.mean()
        
        print(f'{prefix} AUROC mean {aurocMean:0.4f}')
        
        for i, class_id in enumerate(class_ids):
            
            print(f'{self.test_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.4f}')
    def print_base_indicator (self, precisionIndividual, indicator ,prefix='val'):
        
        precisionMean = precisionIndividual.mean()
        
        print(f'{prefix} {indicator}  {precisionMean:0.4f}')
        
        
        
        
    def pred_matrix_to_zero_one(self,outPRED):
            
            result = torch.zeros_like(outPRED)
            
            result[(outPRED >= 0.5)] = 1
            return result


    def compute_F1s_threshold(self, gt, pred, threshold, n_class=12):
        gt_np = gt
        pred_np = pred

        F1s = []
        F1s.append('F1s')
        for i in range(n_class):
            pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
            pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
            F1s.append(f1_score(gt_np[:, i], pred_np[:, i], average='binary'))  
        mean_f1 = np.mean(np.array(F1s[1:]))
        F1s.append(mean_f1)
        return F1s

    def compute_Accs_threshold(self, gt, pred, threshold, n_class=12):
        gt_np = gt
        pred_np = pred
        Accs = []
        Accs.append('Accs')
        for i in range(n_class):
            pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
            pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
            Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
        mean_accs = np.mean(np.array(Accs[1:]))
        Accs.append(mean_accs)
        return Accs

    def compute_mccs_threshold(self, gt, pred, threshold, n_class=12):
        gt_np = gt
        pred_np = pred
        mccs = []
        mccs.append('mccs')
        for i in range(n_class):
            pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
            pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
            mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
        mean_mccs = np.mean(np.array(mccs[1:]))
        mccs.append(mean_mccs)
        return mccs

    def compute_mccs(self, gt, pred, n_class=14):
        
        gt_np = gt
        pred_np = pred
        select_best_thresholds = []
        best_mcc = 0.0

        for i in range(n_class):
            select_best_threshold_i = 0.0
            best_mcc_i = 0.0
            for threshold_idx in range(len(pred)):
                pred_np_ = pred_np.copy()
                thresholds = pred[threshold_idx]
                pred_np_[:, i][pred_np_[:, i] >= thresholds[i]] = 1
                pred_np_[:, i][pred_np_[:, i] < thresholds[i]] = 0
                mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
                if mcc > best_mcc_i:
                    best_mcc_i = mcc
                    select_best_threshold_i = thresholds[i]
            select_best_thresholds.append(select_best_threshold_i)
            print(f'第{i}类找到了阈值')
        for i in range(n_class):
            pred_np[:, i][pred_np[:, i] >= select_best_thresholds[i]] = 1
            pred_np[:, i][pred_np[:, i] < select_best_thresholds[i]] = 0
        mccs = []
        mccs.append('mccs')
        for i in range(n_class):
            mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
        mean_mcc = np.mean(np.array(mccs[1:]))
        mccs.append(mean_mcc)
        return mccs, select_best_thresholds
