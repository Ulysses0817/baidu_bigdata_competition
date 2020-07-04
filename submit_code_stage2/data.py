from utils import *
# from self.config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random, pathlib
from HazeRemoval import getRecoverScene
import cv2, pickle

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class PlanetDataset(Dataset):
	def __init__(self, data_df, base_path=False, augument=True, mode="train", dtype="image", config=None, transforms_id=None, dehaze=False, randomdehaze=False):
		
		self.config = config
		self.data_df = data_df.copy()
		self.augument = augument
		self.mode = mode
		self.dtype = dtype
		self.seq_list = self.aug_list()
		self.tid = transforms_id
		self.dehaze = dehaze
		self.randomdehaze = randomdehaze
		
	def __len__(self):
		return len(self.data_df)

	def __getitem__(self, index):
		# index = int(index)
		if self.dtype == "image":
			X = self.read_images(index)
			# X = getRecoverScene(X, refine=True)
			if not self.mode == "test":
				labels = np.array([self.data_df.iloc[index].Target])
				y  = self.data_df.iloc[index].Target#np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
			else:
				y = self.data_df.iloc[index].basename
			if self.augument:
				X = self.augumentor(X)
			#X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])(X)
			if self.mode == "train":
				X = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])(X) #T.RandomCrop(88), T.Pad(padding=32, fill=0), 
			else:
				X = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])(X) #T.CenterCrop(88), 
			return X.float(), y
			
		elif self.dtype == "visit":
			X = self.read_visit(index).transpose((1, 2, 0))/4
			# X = X.transpose((1, 2, 0))/4#np.log(X+1)*255/7.63#
			# visit_l1 = X / (np.sum(X, 0, keepdims=True) + 1e-8)
			# visit_log = np.log1p(X)
			# X = np.concatenate([X/16, visit_log], 2)
			# visit, max, min = norm(visit)
			# visit_0_1 = visit.astype(np.float) / 255.0
			# visit_std = std(visit)
			# mean = np.mean(X.)
			if not self.mode == "test":
				labels = np.array([self.data_df.iloc[index].Target])
				y  = self.data_df.iloc[index].Target#np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
			else:
				y = self.data_df.iloc[index].basename
			# X = T.Compose([T.ToTensor(), T.Normalize(mean = [0.00308299, 0.00330007, 0.00313859, 0.00304339, 0.00304015, 0.00300855, 0.00299008],
						# std = [0.00632953, 0.00655441, 0.00647438, 0.0063527 , 0.00614851, 0.00587151, 0.00592664])])(X)#, T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])mean = [0.013748469], std = [0.026731109]
			X = T.Compose([T.ToTensor(), T.Normalize(mean = [0.00308626], std = [0.00624236])])(X)
			return X.float(), y                     #mean = [0.0123450], std = [0.0249694] #mean = [0.013748469], std = [0.026731109][0.02531813, 0.02621766, 0.02589754, 0.02541081, 0.02459403, 0.02348605, 0.02370658]
			
		elif self.dtype == "mixed":
			X_img = self.read_images(index)
			X_visit = self.read_visit(index).transpose((1, 2, 0))/16#.reshape(24, 182, 1)
			if not self.mode == "test":
				labels = np.array([self.data_df.iloc[index].Target])
				y  = self.data_df.iloc[index].Target#np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
			else:
				y = self.data_df.iloc[index].basename
			if self.augument and isinstance(self.tid, type(None)):
				X_img = self.augumentor(X_img)
			elif not isinstance(self.tid, type(None)):
				X_img = self.seq_list[self.tid].augment_image(X_img)
			X_img = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])(X_img)
			X_visit = T.Compose([T.ToTensor(), T.Normalize(mean = [0.00308299, 0.00330007, 0.00313859, 0.00304339, 0.00304015, 0.00300855, 0.00299008],
						std = [0.00632953, 0.00655441, 0.00647438, 0.0063527 , 0.00614851, 0.00587151, 0.00592664])])(X_visit)
			
			return X_img.float(), X_visit.float(), y
			
		elif self.dtype == "mixed_v1":
			X_img = self.read_images(index)
			X_visit = self.read_visit(index).reshape(24, 182, 1)/16
			# visit_log = np.log1p(X_visit)
			# X_visit = np.concatenate([X_visit/16, visit_log], 2)
			# X_visit = np.log(X_visit+1)*255/7.63 #/4#.reshape(24, 182, 1)
			if not self.mode == "test":
				labels = np.array([self.data_df.iloc[index].Target])
				y  = self.data_df.iloc[index].Target#np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
			else:
				y = self.data_df.iloc[index].basename
			if self.augument and isinstance(self.tid, type(None)):
				X_img = self.augumentor(X_img)
			elif not isinstance(self.tid, type(None)):
				X_img = self.seq_list[self.tid].augment_image(X_img)
			X_img = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])(X_img)
			X_visit = T.Compose([T.ToTensor(), T.Normalize(mean = [np.double(0.00308626)], std = [np.double(0.00624236)])])(X_visit)
														# mean = [0.220488723], std = [0.191964952]  mean = [0.013748469], std = [0.026731109]
			return X_img.float(), X_visit.float(), y
		
		## TO DO...
		elif self.dtype == "tri_mode":
			"""
			add handcrafted features
			"""
			X_img = self.read_images(index)
			X_visit = self.read_visit(index).transpose((1, 2, 0))/4#.reshape(24, 182, 1)
			X_handfetures = self.read_handcrafted_features(index)
			if not self.mode == "test":
				labels = np.array([self.data_df.iloc[index].Target])
				y  = self.data_df.iloc[index].Target#np.eye(self.config.num_classes, dtype=np.float)[labels].sum(axis=0)
			else:
				y = self.data_df.iloc[index].basename
			if self.augument:
				X_img = self.augumentor(X_img)
			X_img = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])])(X_img)
			X_visit = T.Compose([T.ToTensor(), T.Normalize(mean = [0.01233194, 0.01320027, 0.01255435, 0.01217356, 0.01216061, 0.0120342 , 0.0119603],
						std = [0.02531813, 0.02621766, 0.02589754, 0.02541081, 0.02459403, 0.02348605, 0.02370658])])(X_visit)
			X_handfetures = ""
			return X_img.float(), X_visit.float(), X_handfetures.float(), X_y
			
	def read_images(self,index):
		row = self.data_df.iloc[index]
		filename = str(row.Id_x)#.replace("train", "train_dehaze").replace("test", "test_dehaze")
		bool = False
		if self.randomdehaze:
			filename = random.choice([filename, filename.replace("train", "train_dehaze").replace("test", "test_dehaze")])
			bool = True
		elif self.dehaze:
			assert(bool == False)
			filename = filename.replace("train", "train_dehaze").replace("test", "test_dehaze")
		images = np.array(Image.open(filename))
		if self.config.img_height == 100:
			return images
		else:
			return cv2.resize(images,(self.config.img_weight,self.config.img_height))

	def read_visit(self,index):
		row = self.data_df.iloc[index]
		filename = str(row.Id_y)
		varray = np.load(filename)
		with open(filename, "rb") as f:
			varray = pickle.load(f)[0]
		return varray.astype("float32")
		
	def read_handcrafted_features(self,index):
		row = self.fe_df.iloc[index]
		v_feature = row.values
		return v_feature
		

	def aug_list(self):
		flip = [iaa.Noop(), iaa.Fliplr(1)]
		aff = [ iaa.Affine(rotate=0),
				iaa.Affine(rotate=90),
				iaa.Affine(rotate=180),
				iaa.Affine(rotate=270)]

		seq_list = []
		for i in range(2):
			for j in range(4):
				img_aug = iaa.Sequential([flip[i], aff[j]])
				seq_list.append(img_aug)
		return seq_list
			
	def augumentor(self, image):
		# augment_img = iaa.Sequential([
			# iaa.OneOf([
				# iaa.Affine(rotate=90),
				# iaa.Affine(rotate=180),
				# iaa.Affine(rotate=270),
				# iaa.Affine(rotate=0),
				# iaa.Fliplr(1),
				# iaa.Flipud(1),
				# # iaa.ContrastNormalization(alpha=1.2, per_channel=False)
			# ])], random_order=True)
		image_aug = random.choice(self.seq_list).augment_image(image)
		
		return image_aug
	
	def tta(self):
		pass