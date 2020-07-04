import numpy as np
from functools import partial
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange, tqdm
import sys
import glob
import torch
import swifter
from torch import nn
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from attention import MultiHeadAttention, BiATT, EncoderLayer
from torch.optim.optimizer import Optimizer
import torchvision
from torchvision import models
import pretrainedmodels
from pretrainedmodels.models import *
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2
import pickle
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class MultiModalDataset(Dataset, ):
	def __init__(self,images_df, base_path,vis_path,augument=True,mode="train",mixed_up=False,label_smooth=False,TTA=False):#,magic
		if not isinstance(base_path, pathlib.Path):
			base_path = pathlib.Path(base_path)
		if not isinstance(vis_path, pathlib.Path):
			vis_path = pathlib.Path(vis_path)
		if isinstance(images_df, list):
			self.images_df_new = images_df[0].copy() #csv
			self.images_df_old = images_df[1].copy() #csv
		else:
			print("test mode", len(images_df))
			self.images_df = images_df
		# self.magic = magic
		# if not mode == "test":
		#     assert len(images_df) == len(self.magic),print(len(images_df),len(self.magic_trains))
		# else:
		#     assert len(images_df) == len(self.magic),print(len(images_df),len(self.magic_trains))

		self.augument = augument
		self.mixed_up = mixed_up
		self.TTA = TTA
		if self.mixed_up:
			random.seed(2019)
			self.mixup_inxs = list(range(0, self.images_df.shape[0], 1))
			random.shuffle(self.mixup_inxs)
		self.label_smooth = label_smooth
		self.label_smooth_lambda = 0.1
		self.vis_path = vis_path #vist npy path
		
		if not mode == "test":
			if isinstance(images_df, list):
				self.images_df_new.Id = self.images_df_new.Id.apply(lambda x:
														glob.glob("../data/train/*/" + x + ".jpg")[0])
				self.images_df_old.Id = self.images_df_old.Id.apply(lambda x:
														glob.glob("../data_old/train_image/train/%s/"%x.split("_")[1] + x + ".jpg")[0])
			else:
				self.images_df.Id = self.images_df.Id.apply(lambda x:
														glob.glob("../data/train/*/" + x + ".jpg")[0])
		else:
			if isinstance(images_df, list):
				self.images_df_new.Id = self.images_df_new.Id.apply(lambda x:
														glob.glob("../data/test/" + str(x).zfill(6) + ".jpg")[0])
				self.images_df_old.Id = self.images_df_old.Id.apply(lambda x:
														glob.glob("../data_old/test_image/test/" + str(x).zfill(6) + ".jpg")[0])
			else:
				self.images_df.Id = self.images_df.Id.apply(lambda x:
														glob.glob("../data/test/" + str(x).zfill(6) + ".jpg")[0])
		if isinstance(images_df, list):
			self.images_df = pd.concat([self.images_df_new, self.images_df_old], axis=0)
		assert(self.images_df.shape[-1]==2)
		# self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / str(x).zfill(6))
		self.mode = mode
		# ['len'] + q + p + ['sum_div_len', 'sum', 'argmax_p', 'argmax_q']  # ,'maxpq','argmin_p','argmin_q','minpq'
		self.feature = ['hour_'+str(i) for i in range(24)] + \
					   ['day_'+str(i) for i in range(7)] + \
					   ['len'] + \
					   ['sum_div_len', 'sum', 'argmax_p', 'argmax_q', 'maxpq','argmin_p','argmin_q','minpq']

	def __len__(self):
		return len(self.images_df)

	def __getitem__(self,index):
		X, filename = self.read_images(index)

		visit = self.read_npy(index)
		visit=visit.reshape((26*7, 24)).transpose(1,0)
		#magic = self.magic.iloc[index][self.feature]
		# Id = self.magic.iloc[index]['Id']
		# assert filename == Id, print(filename, Id)
		if not self.mode == "test" and self.augument:
			X = self.augumentor(X)
		if self.mode == "test":
			if self.TTA:
				X = [X] + [x for x in self.TTA_func(X)]
			else:
				X = [X]
		if self.mixed_up:
			mixup_inx = self.mixup_inxs[index]
			mix_data,_ = self.read_images(mixup_inx)
			if self.augument:
				mix_data = self.augumentor(mix_data)
			X = (X + mix_data) / 2
			X = T.Compose([T.ToTensor()])(X)
		if not self.mixed_up:
			X = [T.Compose([T.ToPILImage(),T.ToTensor()])(x) for x in X]

		if not self.mode == "test":
			y = self.images_df.iloc[index].Target
			y = np.eye(9)[y]
			if self.label_smooth:
				y = y * (1 - self.label_smooth_lambda) + \
						(1 - y) * self.label_smooth_lambda / 8
			# print('label_smooth', index, y)
			if self.mixed_up:
				mixup_inx = self.mixup_inxs[index]
				mixup_y = self.images_df.iloc[mixup_inx].Target
				mixup_y = np.eye(9)[mixup_y]
				if self.label_smooth:
					mixup_y = mixup_y * (1 - self.label_smooth_lambda) + \
						(1 - mixup_y) * self.label_smooth_lambda / 8
				old_y = y
				y = (y + mixup_y) / 2
			# print('mixed_up', index, old_y, mixup_inx, mixup_y, y)
		else:
			y = os.path.basename(self.images_df.iloc[index].Id)
		# print(X.shape)

		# visit=T.Compose([T.ToTensor()])(visit)
		# print(magic, magic.shape)
		visit = torch.Tensor(visit)
		# magic = torch.Tensor(magic)
		# print(visit.shape, magic.shape)
		if not self.mode == "test":
			return [x.float() for x in X], (visit.float(), ), y#magic.float()
		else:
			return [x.float() for x in X], (visit.float(), ), y#magic.float()

			
			
	def TTA_func(self,image):
		tta = []
		augment_img1 = iaa.Sequential([iaa.Fliplr(1.0)])
		tta.append(augment_img1.augment_image(image))

		augment_img2 = iaa.Sequential([iaa.Flipud(1.0)])
		tta.append(augment_img2.augment_image(image))

		augment_img3 = iaa.Sequential([iaa.Affine(rotate=90)])
		tta.append(augment_img3.augment_image(image))

		augment_img4 = iaa.Sequential([iaa.Affine(rotate=180)])
		tta.append(augment_img4.augment_image(image))

		augment_img5 = iaa.Sequential([iaa.Affine(rotate=270)])
		tta.append(augment_img5.augment_image(image))

		augment_img6 = iaa.Sequential([iaa.GaussianBlur((0, 3.0))])
		tta.append(augment_img6.augment_image(image))

		augment_img7 = iaa.Sequential([iaa.AverageBlur(k=(2, 7))])
		tta.append(augment_img7.augment_image(image))

		augment_img8 = iaa.Sequential([iaa.MedianBlur(k=(3, 11))])
		tta.append(augment_img8.augment_image(image))
		
		return tta

	def read_images(self,index):
		row = self.images_df.iloc[index]
		filename = str(row.Id)
		#print(filename)
		images = cv2.imread(filename)
		images = cv2.resize(images, config.target_size)
		return images, filename.split('/')[-1]

	def read_npy(self,index):
		row = self.images_df.iloc[index]
		filename = os.path.basename(str(row.Id))
		# pth=os.path.join(self.vis_path.absolute(),filename.split('.jpg')[0]+'.npy')
		# visit=np.load(pth)
		if "data_old" in str(row.Id):
			if "train" in str(row.Id):
				with open(os.path.join("../data_old/npy2/train_visit/",filename.split('.jpg')[0]+'.pkl'), 'rb') as f:
					(visit, _) = pickle.load(f)	
			else:
				with open(os.path.join("../data_old/npy2/test_visit/",filename.split('.jpg')[0]+'.pkl'), 'rb') as f:
					(visit, _) = pickle.load(f)
		else:
			with open(os.path.join(self.vis_path.absolute(),filename.split('.jpg')[0]+'.pkl'), 'rb') as f:
				(visit, _) = pickle.load(f)	
		return visit

	def augumentor(self,image):
		augment_img = iaa.Sequential([
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.SomeOf((0,4),[
				iaa.Affine(rotate=90),
				iaa.Affine(rotate=180),
				iaa.Affine(rotate=270),
				iaa.Affine(shear=(-16, 16)),
			]),
			iaa.OneOf([
					iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
					iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
					iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
				]),
			#iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
			], random_order=True)

		image_aug = augment_img.augment_image(image)
		return image_aug


class _LRScheduler(object):
	def __init__(self, optimizer, last_epoch=-1):
		if not isinstance(optimizer, Optimizer):
			raise TypeError('{} is not an Optimizer'.format(
				type(optimizer).__name__))
		self.optimizer = optimizer
		if last_epoch == -1:
			for group in optimizer.param_groups:
				group.setdefault('initial_lr', group['lr'])
		else:
			for i, group in enumerate(optimizer.param_groups):
				if 'initial_lr' not in group:
					raise KeyError("param 'initial_lr' is not specified "
								   "in param_groups[{}] when resuming an optimizer".format(i))
		self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
		self.step(last_epoch + 1)
		self.last_epoch = last_epoch

	def get_lr(self):
		raise NotImplementedError

	def step(self, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
		self.last_epoch = epoch
		for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
			param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
	def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
		self.T_max = T_max
		self.eta_min = eta_min
		self.optimizer = optimizer
		super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [self.eta_min + (base_lr - self.eta_min) *
				(1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
				for base_lr in self.base_lrs]
	
	def _reset(self, epoch, T_max):
		"""
		Resets cycle iterations.
		Optional boundary/step size adjustment.
		"""
		return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)



class FCViewer(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

	
'''Dual Path Networks in PyTorch.'''
class Bottleneck(nn.Module):
	def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
		super(Bottleneck, self).__init__()
		self.out_planes = out_planes
		self.dense_depth = dense_depth
		self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
		self.bn2 = nn.BatchNorm2d(in_planes)
		self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

		self.shortcut = nn.Sequential()
		if first_layer:
			self.shortcut = nn.Sequential(
				nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_planes+dense_depth)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		x = self.shortcut(x)
		d = self.out_planes
		out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
		out = F.relu(out)
		return out


# cnn + gru
class VisitNet_v5(nn.Module):
	def __init__(self):
		super(VisitNet_v5, self).__init__()
		self.gru_1 = nn.GRU(182, 128, bidirectional=True, batch_first=True, dropout=0.2)
		self.gru_2 = nn.GRU(256, 256, bidirectional=True, batch_first=True, dropout=0.2)
		self.slf_attn = MultiHeadAttention(
			8, 512, 64, 64, dropout=0.1)
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(512, 128, kernel_size=h, stride=1, padding=1, bias=False),
						  nn.ReLU(),
						  nn.MaxPool1d(kernel_size=24 - h + 1)
						  )
			for h in [1, 3, 5, 7]
		])
		self.linear = nn.Linear(512, 256)

	def forward(self, x):
		out, _ = self.gru_1(x)
		# print(out.shape)
		out, _ = self.gru_2(out)
		out, dec_slf_attn = self.slf_attn(out, out, out, mask=None)
		out = out.permute(0,2,1)
		# print(out.shape)
		out = [conv(out) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		out = self.linear(out)
		return out


# transformer
class VisitNet_v4(nn.Module):
	def __init__(self):
		super(VisitNet_v4, self).__init__()
		self.gru = nn.GRU(24, 128, bidirectional=True, batch_first=True, dropout=0.1)
		self.layer_stack = nn.ModuleList([
			EncoderLayer(256, 1024, 8, 64, 64, dropout=0.1)
			for _ in range(3)])
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(256, 128, kernel_size=h, stride=1, padding=1, bias=False),
						  nn.ReLU(),
						  nn.MaxPool1d(kernel_size=182 - h + 1)
						  )
			for h in [1, 3, 5, 7]
		])
		self.linear = nn.Linear(512, 256)

	def forward(self, x):
		out, _ = self.gru(x)
		for enc_layer in self.layer_stack:
			out, enc_slf_attn = enc_layer(out,)
		out = out.permute(0,2,1)
		out = [conv(out) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		out = self.linear(out)
		# print(out.shape)
		return out

# biatt
class VisitNet_v3(nn.Module):
	def __init__(self):
		super(VisitNet_v3, self).__init__()
		self.gru_1 = nn.GRU(128, 128, bidirectional=True, batch_first=True, dropout=0.2)
		self.gru_2 = nn.GRU(256, 128, bidirectional=True, batch_first=True, dropout=0.2)
		self.slf_attn = MultiHeadAttention(
			8, 256, 64, 64, dropout=0.1)
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(182, 32, kernel_size=3, stride=1, padding=1, bias=False),
						  nn.ReLU(),
						  )
			for h in [1, 3, 5, 7]
		])

	def forward(self, x):
		out = x.permute(0,2,1)
		out = [conv(out) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.permute(0,2,1)
		out, _ = self.gru_1(out)
		out, _ = self.gru_2(out)
		out, dec_slf_attn = self.slf_attn(out, out, out, mask=None)

		return out

# cnn + gru
class VisitNet_v2(nn.Module):
	def __init__(self):
		super(VisitNet_v2, self).__init__()
		self.gru_1 = nn.GRU(128, 128, bidirectional=True, batch_first=True, dropout=0.2)
		self.gru_2 = nn.GRU(256, 256, bidirectional=True, batch_first=True, dropout=0.2)
		self.slf_attn = MultiHeadAttention(
			8, 512, 64, 64, dropout=0.1)
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
						  nn.ReLU(),
						  )
			for h in [1, 3, 5, 7]
		])
		self.linear = nn.Linear(512, 256)

	def forward(self, x):
		out = x.permute(0,2,1)
		out = [conv(out) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.permute(0,2,1)
		out, _ = self.gru_1(out)
		out, _ = self.gru_2(out)
		out, dec_slf_attn = self.slf_attn(out, out, out, mask=None)
		out = out.permute(0,2,1)
		out = F.adaptive_avg_pool1d(out, 1)
		# # print(out.shape)
		out = out.view(-1, out.size(1))
		out = self.linear(out)
		return out

# gru + cnn
class VisitNet(nn.Module):
	def __init__(self):
		super(VisitNet, self).__init__()
		self.gru_1 = nn.GRU(24, 128, bidirectional=True, batch_first=True, dropout=0.2)
		self.gru_2 = nn.GRU(256, 256, bidirectional=True, batch_first=True, dropout=0.2)
		self.slf_attn = MultiHeadAttention(
			8, 512, 64, 64, dropout=0.1)
		self.convs = nn.ModuleList([
			nn.Sequential(nn.Conv1d(512, 128, kernel_size=h, stride=1, padding=1, bias=False),
						  nn.ReLU(),
						  nn.MaxPool1d(kernel_size=182 - h + 1)
						  )
			for h in [1, 3, 5, 7]
		])
		self.linear = nn.Linear(512, 256)

	def forward(self, x):
		out, _ = self.gru_1(x)
		# print(out.shape)
		out, _ = self.gru_2(out)
		out, dec_slf_attn = self.slf_attn(out, out, out, mask=None)
		out = out.permute(0,2,1)
		out = [conv(out) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		out = self.linear(out)
		return out


class MultiModalNet(nn.Module):
	def __init__(self, drop):
		super(MultiModalNet, self).__init__()

		img_model = se_resnext50_32x4d(pretrained="imagenet") #models.resnet50(True) # #pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
		self.visit_model = VisitNet_v5() #VisitNet()#

		self.img_encoder = list(img_model.children())[:-2]
		self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
		self.img_encoder = nn.Sequential(*self.img_encoder)

		self.img_fc = nn.Sequential(FCViewer(),
									nn.Dropout(drop),
									nn.Linear(img_model.last_linear.in_features, 256))
									#nn.Linear(img_model.fc.in_features, 256)) #res50
		self.magic_linear = nn.Sequential(FCViewer(), nn.BatchNorm1d(40),
										  nn.Linear(40, 64))#, nn.Linear(1, 4)
		self.cls = nn.Sequential(nn.ReLU(), nn.Linear(512, 128),
								 nn.ReLU(), nn.Linear(128, config.num_classes))

	def forward(self, x_img, x_vis):
		x_img = self.img_encoder(x_img)
		x_img = self.img_fc(x_img)
		x_vis = self.visit_model(x_vis)
		# x_magic = self.magic_linear(magic)
		# print(x_magic.shape)
		# fc 2 network
		# print(x_vis.shape, x_magic.shape)
		x_cat = torch.cat((x_img,x_vis),1)
		# x_cat = x_vis
		x_cat = self.cls(x_cat)
		return x_cat#F.log_softmax(x_cat, dim=1)

# biatt
class MultiModalNet_v2(nn.Module):
	def __init__(self, drop):
		super(MultiModalNet_v2, self).__init__()
		#50 1024 16 16
		#34 256 16 16
		#18 256 16 16 .vgg16(True)
		img_model = models.resnet34(True)#pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
		self.visit_model = VisitNet_v3()
		# print(len(list(img_model.children())))
		# for l in list(img_model.children()):
		#     print(l)
		self.img_encoder = list(img_model.children())[:-3]
		# self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
		self.img_encoder = nn.Sequential(*self.img_encoder)
		self.magic_linear = nn.Sequential(FCViewer(), nn.BatchNorm1d(40),
										  nn.Linear(40, 64))
		self.cls = nn.Sequential(nn.Dropout(0.3),nn.ReLU(),nn.Linear(1024+64,128),
								 nn.ReLU(),nn.Linear(128,config.num_classes))
		self.biatt = BiATT(128)

	def forward(self, x_img,x_vis, magic):
		x_img = self.img_encoder(x_img)
		# print(x_img.shape)
		x_img = x_img.view(-1, 256, 16*16)
		x_img = x_img.permute(0,2,1) #N 16 16 256
		# print(x_img.shape)
		x_vis = self.visit_model(x_vis) #N 182 256
		# print(x_vis.shape)

		# biatt 2 networfk
		out = self.biatt(x_img, x_vis) #N 256 1024

		out = out.permute(0,2,1)
		out = F.adaptive_avg_pool1d(out, 1)
		# print(out.shape)
		x_cat = out.view(-1, out.size(1))
		# print(x_cat.shape)
		x_magic = self.magic_linear(magic)
		x_cat = torch.cat((x_cat, x_magic), 1)
		x_cat = self.cls(x_cat)
		return F.log_softmax(x_cat, dim=1)
