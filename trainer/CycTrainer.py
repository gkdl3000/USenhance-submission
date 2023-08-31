#!/usr/bin/python3
import logging
import itertools
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import json
import torch.nn as nn
from .utils import Logger, Resize, tensor2image
from .datasets import ImageDataset,ValDataset,InferDataset
from .CycleGan import *
from torchvision.transforms import ColorJitter, RandomAffine,Normalize,ToTensor, RandomHorizontalFlip, RandomCrop
from skimage import metrics
import numpy as np
import cv2
from scipy.signal import fftconvolve
from .loss import PerceptualLoss

SEED = 1234
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cyc_Trainer():
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.num_clus = 50
        self.cluster_A_dict = dict()
        self.cluster_B_dict = dict()

        if not os.path.exists(self.config["save_root"]):
            os.makedirs(self.config["save_root"])
        self.log = logging.getLogger()

        self.log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.log.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(config['save_root'], 'log.txt'))
        file_handler.setFormatter(formatter)
        self.log.addHandler(file_handler)
        self.relu = nn.ReLU()
  
        self.netG_A2B = nn.DataParallel(Generator(config['input_nc'], config['output_nc']),device_ids=self.config['device_ids']).to(device)
        self.netG_B2A = nn.DataParallel(Generator(config['input_nc'], config['output_nc']),device_ids=self.config['device_ids']).to(device)
        self.netD_A = nn.DataParallel(Discriminator(config['input_nc']),device_ids=self.config['device_ids']).to(device)
        self.netD_B = nn.DataParallel(Discriminator(config['input_nc']),device_ids=self.config['device_ids']).to(device)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

            
        if config['pretrain']:
            if os.path.isfile(self.config['save_root'] + 'netG_A2B.pth'):
                self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
            if os.path.isfile(self.config['save_root'] + 'netG_B2A.pth'):
                self.netG_B2A.load_state_dict(torch.load(self.config['save_root'] + 'netG_B2A.pth'))
            if os.path.isfile(self.config['save_root'] + 'netD_A.pth'):
                self.netD_A.load_state_dict(torch.load(self.config['save_root'] + 'netD_A.pth'))
            if os.path.isfile(self.config['save_root'] + 'netD_B.pth'):
                self.netD_B.load_state_dict(torch.load(self.config['save_root'] + 'netD_B.pth'))


        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        # self.device = 'cuda' if config['cuda'] else 'cpu'
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['cropsize'], config['cropsize']).to(device)
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['cropsize'], config['cropsize']).to(device)
        self.input_A_valid = Tensor(1, config['input_nc'], config['size'], config['size']).to(device)
        self.input_B_valid = Tensor(1, config['output_nc'], config['size'], config['size']).to(device)
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)
        
        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.adv_loss = torch.nn.BCELoss()

        self.ploss = PerceptualLoss(use_input_norm=False)
        self.ploss.vgg = self.ploss.vgg.to(device)
        self.ploss.lossfn = self.ploss.lossfn.to(device)

        # Dataset loader
        transforms_1 = [RandomAffine(degrees=2,translate=[0.02, 0.02],scale=[1-0.02, 1+0.02]),
                   ToTensor(),
                   Normalize(0.5,0.5),
                   Resize(size_tuple = (config['size'], config['size']))
                   ]
        if config['finetune']:
            transforms_2 = [RandomCrop(size = (config['cropsize'], config['cropsize'])),
                            ColorJitter(brightness = 0.05),
                            RandomHorizontalFlip(0.5), 
                            ToTensor(),
                            Normalize(0.5,0.5),
                            Resize(size_tuple = (config['cropsize'], config['cropsize']))
                    ]
        else:
            transforms_2 = [RandomAffine(degrees=5,translate=[0.05, 0.05],scale=[0.9, 1.1]),
                            RandomCrop(size = (config['cropsize'], config['cropsize'])),
                            ColorJitter(brightness = 0.05),
                            RandomHorizontalFlip(0.5), 
                            ToTensor(),
                            Normalize(0.5,0.5),
                            Resize(size_tuple = (config['cropsize'], config['cropsize']))
                    ]
        
        self.dataloader = DataLoader(ImageDataset(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last=True)

        val_transforms = [ToTensor(),
                          Normalize(0.5,0.5),
                          Resize(size_tuple = (config['size'], config['size']))
                          ]
        
        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms),
                                batch_size=1, shuffle=False, num_workers=config['n_cpu'])
        self.trainloader_to_infer = DataLoader(ValDataset(config['dataroot'], transforms_ =val_transforms),
                                batch_size=1, shuffle=False, num_workers=config['n_cpu'])

    
    def train(self):
        ###### Training ######
        # Loss plot
        if self.config['visdom']:
            self.visdom_logger = Logger(self.config['name'],
                                self.config['port'],
                                self.config['n_epochs']-self.config['epoch'], 
                                len(self.dataloader))   

        # define best score
        best_score = 0.0
        if os.path.isfile(os.path.join(self.config['save_root'], 'best_score.npy')):
            best_score = np.load(os.path.join(self.config['save_root'], 'best_score.npy'))
            best_score = float(best_score)
            self.log.info('best score: {}'.format(best_score))

        # setting k-means 
        num_clus = self.num_clus
        km_A = KMeans(n_clusters=num_clus, n_init = 10, random_state=SEED)
        km_B = KMeans(n_clusters=num_clus, n_init = 10, random_state=SEED)

        # define cluster info
        cluster_A_savepath = os.path.join(self.config['save_root'], 'cluster_A.json')
        cluster_B_savepath = os.path.join(self.config['save_root'], 'cluster_B.json')
        if os.path.isfile(cluster_A_savepath):
            with open(cluster_A_savepath, 'r') as f:
                cluster_data = json.load(f)
                for key, value in cluster_data.items():
                    self.cluster_A_dict[key] = int(value)
                self.log.info('Clustering A Info Loaded.')
        if os.path.isfile(cluster_B_savepath):
            with open(cluster_B_savepath, 'r') as f:
                cluster_data = json.load(f)
                for key, value in cluster_data.items():
                    self.cluster_B_dict[key] = int(value)
                self.log.info('Clustering B Info Loaded.')

        # clustering
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            if epoch%10==0:
                # K-means clustering and pseudo labeling
                with torch.no_grad():
                    self.log.info('Feature K-means Clustering and Pseudo Labeling..')
                    cluster_A_savepath = os.path.join(self.config['save_root'], 'cluster_A.json')
                    cluster_B_savepath = os.path.join(self.config['save_root'], 'cluster_B.json')
                    
                    if epoch>self.config['clustering_epochs']-1: # if epoch is large, stop k-means clustering
                        # load cluster info
                        if os.path.isfile(cluster_A_savepath):
                            with open(cluster_A_savepath, 'r') as f:
                                cluster_data = json.load(f)
                                for key, value in cluster_data.items():
                                    self.cluster_A_dict[key] = int(value)
                                self.log.info('Clustering A Info Loaded.')
                        if os.path.isfile(cluster_B_savepath):
                            with open(cluster_B_savepath, 'r') as f:
                                cluster_data = json.load(f)
                                for key, value in cluster_data.items():
                                    self.cluster_B_dict[key] = int(value)
                                self.log.info('Clustering B Info Loaded.')

                    else: # do k-means clustering
                        np_features_A = []
                        np_features_B = []
                        imgnames = []
                        num_of_diff_A = 0
                        num_of_diff_B = 0

                        # inference model and get feature vectors of data
                        for i, batch in enumerate(self.trainloader_to_infer): 
                            imgnames.append(batch['imgname'][0])

                            real_A = Variable(self.input_A_valid.copy_(batch['A']))
                            np_arr_A = self.netG_A2B.module.get_info_feature(real_A)

                            np_features_A.append(np_arr_A)
                            if batch['imgname'][0] not in self.cluster_A_dict.keys():
                                self.cluster_A_dict[batch['imgname'][0]]=0

                            real_B = Variable(self.input_B_valid.copy_(batch['B']))
                            np_arr_B = self.netG_B2A.module.get_info_feature(real_B)
                            np_features_B.append(np_arr_B)
                            if batch['imgname'][0] not in self.cluster_B_dict.keys():
                                self.cluster_B_dict[batch['imgname'][0]]=0
                        
                        np_features_A = np.array(np_features_A)
                        np_features_B = np.array(np_features_B)

                        # k-means fit
                        km_A.fit(np_features_A)
                        km_B.fit(np_features_B)
                        self.log.info('A clusters: {}'.format(np.unique(km_A.labels_, return_counts=True)))
                        self.log.info('B clusters: {}'.format(np.unique(km_B.labels_, return_counts=True)))

                        # cluster number re-labeling 
                        new_A_labels = []
                        new_B_labels = []
                        if epoch<self.config['epoch']+10: # condition for retraining task
                            new_A_labels = km_A.labels_
                            previous_A_labelset = self.make_labelset(km_A, num_clus)
                            new_B_labels = km_B.labels_
                            previous_B_labelset = self.make_labelset(km_B, num_clus)
                        
                        else:  # match previous labels and current labels
                            current_A_labelset = self.make_labelset(km_A, num_clus)
                            _, matching_A_dict = self.label_matching(current_A_labelset, previous_A_labelset, num_clus)
                            for label in km_A.labels_:
                                new_A_labels.append(matching_A_dict[label])
                            current_B_labelset = self.make_labelset(km_B, num_clus)
                            _, matching_B_dict = self.label_matching(current_B_labelset, previous_B_labelset, num_clus)
                            for label in km_B.labels_:
                                new_B_labels.append(matching_B_dict[label])

                        # assign new pseudo label
                        for i in range(len(imgnames)):
                            if self.cluster_A_dict[imgnames[i]]!=new_A_labels[i]:
                                num_of_diff_A+=1
                                self.cluster_A_dict[imgnames[i]]=new_A_labels[i]
                            if self.cluster_B_dict[imgnames[i]]!=new_B_labels[i]:
                                num_of_diff_B+=1
                                self.cluster_B_dict[imgnames[i]]=new_B_labels[i]
                        
                        # save cluster info
                        saving_cluster_A_dict = dict()
                        for key, value in self.cluster_A_dict.items():
                            saving_cluster_A_dict[key] = str(value)
                        with open(cluster_A_savepath, 'w', encoding='utf-8') as f:
                            json.dump(saving_cluster_A_dict, f, indent='\t')

                        saving_cluster_B_dict = dict()
                        for key, value in self.cluster_B_dict.items():
                            saving_cluster_B_dict[key] = str(value)
                        with open(cluster_B_savepath, 'w', encoding='utf-8') as f:
                            json.dump(saving_cluster_B_dict, f, indent='\t')

                        self.log.info('total sample: {}'.format(len(imgnames)))
                        self.log.info('num of changed labels in A: {}'.format(num_of_diff_A))
                        self.log.info('num of changed labels in B: {}'.format(num_of_diff_B))
                    
            # training
            epoch_loss_D_B = 0
            epoch_loss_Total = 0
            epoch_loss_GAN_A2B = 0
            epoch_loss_GAN_B2A = 0
            epoch_loss_vgg = 0
            epoch_loss_aux = 0
            
            for i, batch in enumerate(self.dataloader):

                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                target_A_cls = []
                target_B_cls = []
                for i, imgname in enumerate(batch['imgname']):
                    target_A_cls.append(int(self.cluster_A_dict[imgname]))
                    target_B_cls.append(int(self.cluster_B_dict[imgname]))
                self.target_A_cls = torch.tensor(target_A_cls).to(device)
                self.target_B_cls = torch.tensor(target_B_cls).to(device)

                # set optimizer G
                self.optimizer_G.zero_grad()

                # aux loss A
                fake_B, aux_out = self.netG_A2B(real_A)
                loss_aux = self.cls_loss(aux_out, self.target_A_cls)

                # adv loss
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                # aux loss B
                fake_A, aux_out_rev = self.netG_B2A(real_B)
                loss_aux_rev = self.cls_loss(aux_out_rev, self.target_B_cls)

                # adv loss
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake))

                # Cycle loss
                recovered_A, recovered_aux_out_rev = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                recovered_B, recovered_aux_out = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)
                
                # recovered aux loss
                recovered_loss_aux_rev = self.cls_loss(recovered_aux_out_rev, self.target_B_cls)
                recovered_loss_aux = self.cls_loss(recovered_aux_out, self.target_A_cls)

                # vgg loss
                b,_,h,w = real_A.shape
                fake_B_3ch = fake_B.expand([b,3,h,w])
                real_B_3ch = real_B.expand([b,3,h,w])
                recovered_A_3ch = recovered_A.expand([b,3,h,w])
                real_A_3ch = real_A.expand([b,3,h,w])
                loss_vgg = self.config['VGG_lamda'] * (self.ploss(recovered_A_3ch, real_A_3ch) + 0.2 * self.ploss(fake_B_3ch, real_B_3ch))
                
                # Total loss
                loss_adv = loss_GAN_A2B + loss_GAN_B2A 
                loss_cyc = loss_cycle_ABA + loss_cycle_BAB
                loss_aux_total = self.config['Aux_lamda'] * (loss_aux + recovered_loss_aux + loss_aux_rev + recovered_loss_aux_rev)
                if epoch==0: # for warmup
                    loss_Total = loss_adv + 0.1 * (loss_cyc + loss_aux_total + loss_vgg)
                else:
                    loss_Total = loss_adv + loss_cyc + loss_aux_total + loss_vgg
                
                # identity L1 loss
                real_B_out, _ = self.netG_A2B(real_B)
                real_A_out, _ = self.netG_B2A(real_A)
                id_loss = self.config['Id_lamda'] * (self.L1_loss(real_B_out, real_B) + 0.2 * self.L1_loss(real_A_out, real_A))
                loss_Total += id_loss

                # backprop
                loss_Total.backward()
                nn.utils.clip_grad_norm_(self.netG_A2B.parameters(), self.config['grad_clip'])
                nn.utils.clip_grad_norm_(self.netG_B2A.parameters(), self.config['grad_clip'])
                self.optimizer_G.step()    

                
                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()
                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))
                # Fake loss
                # fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)
                loss_D_A.backward()
                nn.utils.clip_grad_norm_(self.netD_A.parameters(), self.config['grad_clip'])
                self.optimizer_D_A.step()
                
                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real))

                # Fake loss
                # fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake))

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)

                # backprop
                loss_D_B.backward()
                nn.utils.clip_grad_norm_(self.netD_B.parameters(), self.config['grad_clip'])
                self.optimizer_D_B.step()

                # logging
                epoch_loss_Total+=loss_Total/len(self.dataloader)
                epoch_loss_GAN_A2B+=loss_GAN_A2B/len(self.dataloader)
                epoch_loss_GAN_B2A+=loss_GAN_B2A/len(self.dataloader)
                epoch_loss_vgg+=loss_vgg/len(self.dataloader)
                epoch_loss_aux+=loss_aux_total/len(self.dataloader)
                epoch_loss_D_B+=loss_D_B/len(self.dataloader)

                if self.config['visdom']:
                    self.visdom_logger.log({'loss_D_B': loss_D_B,
                                    'loss_Total':loss_Total,
                                    'loss_vgg':loss_vgg, 
                                    'loss_aux':loss_aux_total,
                                    'loss_A2B':loss_GAN_A2B,
                                    'loss_B2A':loss_GAN_B2A},
                                    images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})
            
            self.log.info('Epoch: {}, loss_D_B: {}, loss_Total: {}, loss_A2B: {}, loss_B2A: {}, loss_vgg: {}, loss_aux: {}'.format(epoch, 
                                                                                                                     epoch_loss_D_B, 
                                                                                                                     epoch_loss_Total, 
                                                                                                                     epoch_loss_GAN_A2B,
                                                                                                                     epoch_loss_GAN_B2A, 
                                                                                                                     epoch_loss_vgg,
                                                                                                                     epoch_loss_aux))
            
            # Save models checkpoints
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'netG_B2A.pth')
            torch.save(self.netD_A.state_dict(), self.config['save_root'] + 'netD_A.pth')
            torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'netD_B.pth')

            if (epoch+1)%10==0 or self.config['finetune']:
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + '{:04d}_netG_A2B.pth'.format(epoch))
                torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + '{:04d}_netG_B2A.pth'.format(epoch))
                torch.save(self.netD_A.state_dict(), self.config['save_root'] + '{:04d}_netD_A.pth'.format(epoch))
                torch.save(self.netD_B.state_dict(), self.config['save_root'] + '{:04d}_netD_B.pth'.format(epoch))

            #############val###############
            with torch.no_grad():
                MAE = 0
                PSNR = 0
                SSIM = 0
                NCC = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A_valid.copy_(batch['A']))
                    real_B = Variable(self.input_B_valid.copy_(batch['B']))
                    fake_B, aux_out = self.netG_A2B(real_A)

                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_B = real_B.detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B,real_B)
                    psnr = self.PSNR(fake_B,real_B)
                    ssim = metrics.structural_similarity(fake_B,real_B)
                    ncc = self.normxcorr2(fake_B, real_B, mode="valid")[0,0]

                    MAE += mae
                    PSNR += psnr
                    SSIM += ssim 
                    NCC += ncc
                    
                    num += 1
                
                self.log.info('Val MAE: {}'.format(MAE/num))
                self.log.info('Val PSNR: {}'.format(PSNR/num))
                self.log.info('Val SSIM: {}'.format(SSIM/num))
                self.log.info('Val NCC: {}'.format(NCC/num))

                score = (0.1*PSNR + SSIM + NCC)/num
                if score > best_score:
                    best_score = score
                    np.save(os.path.join(self.config['save_root'], 'best_score.npy'), best_score)
                    self.log.info('best score: {}'.format(best_score))
                    torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'best_netG_A2B.pth')
                    torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'best_netG_B2A.pth')
                    torch.save(self.netD_A.state_dict(), self.config['save_root'] + 'best_netD_A.pth')
                    torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'best_netD_B.pth')
                    
                         
    def test(self,):
        
        alpha = 0.5
        weight1 = torch.load(self.config['model_root'])
        weight2 = torch.load(self.config['model_root2'])
        weight = dict()
        
        for key in weight1.keys():
            weight[key] = alpha * weight1[key] + (1-alpha) * weight2[key]

        self.netG_A2B.load_state_dict(weight)
        os.makedirs(self.config['image_save'],exist_ok=True)
        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            NCC = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A_valid.copy_(batch['A']))
                real_B = Variable(self.input_B_valid.copy_(batch['B']))

                real_A_h = transforms.functional.hflip(real_A)
                fake_B, _ = self.netG_A2B(real_A)
                fake_B_h, _ = self.netG_A2B(real_A_h)
                fake1 = fake_B
                fake2 = transforms.functional.hflip(fake_B_h)
                fake_B = (fake1+fake2)/2
                
                realimage_A = np.squeeze(tensor2image(real_A.data))
                realimage_B = np.squeeze(tensor2image(real_B.data))
                fakeimage_B = np.squeeze(tensor2image(fake_B.data))
                res_image = cv2.hconcat([realimage_A, fakeimage_B, realimage_B])
                cv2.imwrite(os.path.join(self.config['image_save'], str(i).zfill(5)+'.png'), res_image)

                real_B = real_B.detach().cpu().numpy().squeeze()
                fake_B = fake_B.detach().cpu().numpy().squeeze()
                mae = self.MAE(fake_B,real_B)
                psnr = self.PSNR(fake_B,real_B)
                ssim = metrics.structural_similarity(fake_B,real_B)
                ncc = self.normxcorr2(fake_B, real_B, mode="valid")[0,0]
                MAE += mae
                PSNR += psnr
                SSIM += ssim 
                NCC += ncc

                num += 1
            print ('MAE:',MAE/num)
            print ('PSNR:',PSNR/num)
            print ('SSIM:',SSIM/num)
            print ('NCC:',NCC/num)
            with open(os.path.join(self.config['save_root'], 'valid_result.txt'), 'w') as f:
                f.write('MAE: '+ str(MAE/num)+'\n')
                f.write('PSNR: '+ str(PSNR/num)+'\n')
                f.write('SSIM: '+ str(SSIM/num)+'\n')
                f.write('NCC: '+ str(NCC/num)+'\n')


    def inference(self,):
        
        self.netG_A2B.load_state_dict(torch.load(self.config['model_root']))
        val_transforms = [ToTensor(),
                        Normalize(0.5,0.5),
                        Resize(size_tuple = (self.config['size'], self.config['size']))]
        self.infer_data = DataLoader(InferDataset(self.config['infer_dataroot'], transforms_ = val_transforms),
                                batch_size=1, shuffle=False, num_workers=self.config['n_cpu'])
        os.makedirs(self.config['infer_image_save'],exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(self.infer_data):
                imgname = os.path.basename(batch['imgname'][0])

                real_A = Variable(self.input_A_valid.copy_(batch['A']))
                fake_B, _ = self.netG_A2B(real_A)
                
                fakeimage_B = np.squeeze(tensor2image(fake_B.data))

                # print(res_image.shape)
                cv2.imwrite(os.path.join(self.config['infer_image_save'], imgname), fakeimage_B)
    
    def multiinference(self,):

        seed=1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        alpha = 0.5
        weight1 = torch.load(self.config['model_root'])
        weight2 = torch.load(self.config['model_root2'])
        weight = dict()
        
        for key in weight1.keys():
            weight[key] = alpha * weight1[key] + (1-alpha) * weight2[key]

        self.netG_A2B.load_state_dict(weight)
        val_transforms = [ColorJitter(brightness = 0.02),
                          ToTensor(),
                          Normalize(0.5,0.5),
                          Resize(size_tuple = (self.config['size'], self.config['size']))]
        self.infer_data = DataLoader(InferDataset(self.config['infer_dataroot'], transforms_ = val_transforms),
                                batch_size=1, shuffle=False, num_workers=self.config['n_cpu'])
        os.makedirs(self.config['infer_image_save'],exist_ok=True)

        with torch.no_grad():
            
            for i, batch in enumerate(self.infer_data):
                fake_B_arr = []
                for j in range(3):
                    imgname = os.path.basename(batch['imgname'][0])
                    real_A = Variable(self.input_A_valid.copy_(batch['A']))
                    real_A_h = transforms.functional.hflip(real_A)

                    fake_B, _ = self.netG_A2B(real_A)
                    fake_B_h, _ = self.netG_A2B(real_A_h)
                    
                    fake1 = fake_B
                    fake2 = transforms.functional.hflip(fake_B_h)
                    fake_B_j = (fake1+fake2)/2
                    fake_B_arr.append(fake_B_j)
                    
                fake_B = (fake_B_arr[0] + fake_B_arr[1] + fake_B_arr[2])/3
                fakeimage_B = np.squeeze(tensor2image(fake_B.data))

                cv2.imwrite(os.path.join(self.config['infer_image_save'], imgname), fakeimage_B)
    
    
    def label_matching(self, A_labelset, B_labelset, num_clus=50):

        AB_inter_list = []
        for i in range(num_clus):
            A_set = set(A_labelset[i])
            for j in range(num_clus):
                B_set = set(B_labelset[j])
                AB_len = len(A_set.intersection(B_set))
                AB_inter_list.append((i,j,AB_len))

        AB_inter_list.sort(key=lambda x: -x[2])
        
        A_list = [x for x in range(num_clus)]
        B_list = [x for x in range(num_clus)]
        matching_result = []
        matching_dict = dict()

        for Aidx, Bidx, internum in AB_inter_list:
            if Aidx in A_list and Bidx in B_list:
                matching_result.append((Aidx, Bidx, internum))
                matching_dict[Aidx]=Bidx
                A_list.remove(Aidx)
                B_list.remove(Bidx)
                matching_result.sort(key=lambda x: x[0])
        
        return matching_result, matching_dict

    def make_labelset(self, km, num_clus=50):
        label_set = dict()
        for label in range(num_clus):
            label_set[label]=[]
        for i in range(len(km.labels_)):
            label_set[km.labels_[i]].append(i)
        return label_set

    def PSNR(self,fake,real):
       x,y = np.where(real!= -1)# Exclude background
       mse = np.mean(((fake[x,y]+1)/2. - (real[x,y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
    def MAE(self,fake,real):
        x,y = np.where(real!= -1)  # Exclude background
        mae = np.abs(fake[x,y]-real[x,y]).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)
    
    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 

    def normxcorr2(self, template, image, mode="full"):
        """
        Input arrays should be floating point numbers.
        :param template: N-D array, of template or filter you are using for cross-correlation.
        Must be less or equal dimensions to image.
        Length of each dimension must be less than length of image.
        :param image: N-D array
        :param mode: Options, "full", "valid", "same"
        full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
        Output size will be image size + 1/2 template size in each dimension.
        valid: The output consists only of those elements that do not rely on the zero-padding.
        same: The output is the same size as image, centered with respect to the 'full' output.
        :return: N-D array of same dimensions as image. Size depends on mode parameter.
        """

        # If this happens, it is probably a mistake
        if np.ndim(template) > np.ndim(image) or \
                len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
            print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

        template = template - np.mean(template)
        image = image - np.mean(image)

        a1 = np.ones(template.shape)
        # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)
        
        image = fftconvolve(np.square(image), a1, mode=mode) - \
                np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

        # Remove small machine precision errors after subtraction
        image[np.where(image < 0)] = 0

        template = np.sum(np.square(template))
        with np.errstate(divide='ignore',invalid='ignore'): 
            out = out / np.sqrt(image * template)

        # Remove any divisions by 0 or very close to 0
        out[np.where(np.logical_not(np.isfinite(out)))] = 0
        
        return out
    