from torchvision import models
from pretrainedmodels.models import bninception, resnet50, se_resnext50_32x4d, se_resnext101_32x4d, densenet169
# from models.oct_resnet import oct_resnet50
from models.resnet4cifar import resnet56, resnet110, resnet200
from torch import nn
# from config import config
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torch, os

# "bninception-bcelog", "resnet50-bcelog", "se_resnext50_32x4d-bcelog"

def get_net(model_name = None, config = None, img_channels = 3):
	
	if model_name:
		model_name = model_name.split("-")[0]
	
	if model_name == "bninception" :
		print("train %s"%model_name)
		model = bninception(pretrained="imagenet")
		# bnin_dict = model.state_dict()
		# para = bnin_dict['conv1_7x7_s2.weight'].cpu().data.numpy()
		# para = np.concatenate((para, para[:, :1, :, :]), axis=1)
		# model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
		# model.load_state_dict(bnin_dict)
		model.global_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					nn.BatchNorm1d(1024),
					nn.Dropout(0.5),
					nn.Linear(1024, config.num_classes),
				)
		
	elif model_name == "resnet50":
		print("train %s"%model_name)
		model = resnet50(pretrained="imagenet")
		# res_dict = model.state_dict()
		# para = res_dict['conv1.weight'].data.numpy()
		# para = np.concatenate((para,para[:, :1, :, :]), axis=1)
		# res_dict['conv1.weight'] = torch.tensor(para)
		# model.conv1 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		# model.load_state_dict(res_dict)
		model.avgpool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(2048),
					# nn.Dropout(0.5),
					nn.Linear(2048, config.num_classes),
				)

	elif model_name == "octresnet50":
		print("train %s"%model_name)
		model = oct_resnet50(pretrained="imagenet")
		# res_dict = model.state_dict()
		# para = res_dict['conv1.weight'].data.numpy()
		# para = np.concatenate((para,para[:, :1, :, :]), axis=1)
		# res_dict['conv1.weight'] = torch.tensor(para)
		# model.conv1 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		# model.load_state_dict(res_dict)
		# model.avgpool = nn.AdaptiveAvgPool2d(1)
		model.fc = nn.Sequential(
					# nn.BatchNorm1d(2048),
					# nn.Dropout(0.5),
					nn.Linear(2048, config.num_classes),
				)
				
	elif model_name == "seresnext5032x4d":
		print("train %s"%model_name)
		model = se_resnext50_32x4d(pretrained="imagenet")
		# if config.img_weight == 100:
		# print(config.channels)
		if img_channels != 3:
			print("%s for visit"%model_name)
			# model = se_resnext50_32x4d(input_3x3=True, pretrained=None)
			layer0_modules = [
							('conv1', nn.Conv2d(7, 64, 3, stride=2, padding=1,
												bias=False)),
							('bn1', nn.BatchNorm2d(64)),
							('relu1', nn.ReLU(inplace=True)),
							('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
												bias=False)),
							('bn2', nn.BatchNorm2d(64)),
							('relu2', nn.ReLU(inplace=True)),
							('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
												bias=False)),
							('bn3', nn.BatchNorm2d(64)),
							('relu3', nn.ReLU(inplace=True)),
							('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True))
							]
			from collections import OrderedDict
			model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
			# model.layer0.conv1 = nn.Conv2d(img_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
			# model.layer0 =  nn.Sequential(
			# nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)), 
			# nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
			# )
		
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.5),
					# nn.Linear(in_features=2048, out_features=1024, bias=True),
					# nn.ReLU(inplace=True),
					# nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.2),
					nn.Linear(2048, config.num_classes, bias=True),
				)
				
	elif model_name == "seresnext101":
		print("train %s"%model_name)
		model = se_resnext101_32x4d(pretrained="imagenet")
		# if config.img_weight == 100:
		# print(config.channels)
		if img_channels != 3:
			model.layer0 =  nn.Sequential(
			nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)), 
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
			)
		
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.5),
					# nn.Linear(in_features=2048, out_features=1024, bias=True),
					# nn.ReLU(inplace=True),
					# nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.2),
					nn.Linear(2048, config.num_classes, bias=True),
				)

	elif model_name == "densenet169":
		print("train %s"%model_name)
		model = densenet169(pretrained="imagenet")
		# if config.img_weight == 100:
		# print(config.channels)
		# if img_channels != 3:
			# model.layer0 =  nn.Sequential(
			# nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)), 
			# nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
			# )

		model.last_linear = nn.Sequential(
					nn.Linear(1664, config.num_classes, bias=True),
				)
				
	elif model_name == "resnet18":
		print("train %s"%model_name)
		model = resnet18(pretrained="imagenet")
		# model.conv1 = nn.Conv2d(config.channels, 64, kernel_size=(2, 2), stride=(1, 1), padding=(3, 3), bias=False)#
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(2048),
					# nn.Dropout(0.8),
					nn.Linear(512, config.num_classes),
				)
				
	elif model_name == "resnet56":
		print("train %s"%model_name)
		model = resnet56()#pretrained="imagenet"
		checkpoint = torch.load("./models/resnet56.th")
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in checkpoint["state_dict"].items():
			namekey = k[7:] if "module." in k else k# remove `module.`
			if "linear" in namekey:
				namekey = namekey.replace("linear", "last_linear")
			# if "linear" in namekey:
				# namekey = namekey.replace("linear", "last_linear")
			new_state_dict[namekey] = v
			# print(namekey, end="|")
		model.load_state_dict(new_state_dict)
		if img_channels != 3:
			model.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(2048),
					# nn.Dropout(0.8),
					nn.Linear(64, config.num_classes),
				)
				
	elif model_name == "resnet110":
		print("train %s"%model_name)
		model = resnet110()#pretrained="imagenet"
		checkpoint = torch.load("./models/resnet110.th")
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in checkpoint["state_dict"].items():
			namekey = k[7:] if "module." in k else k# remove `module.`
			if "linear" in namekey:
				namekey = namekey.replace("linear", "last_linear")
			# if "linear" in namekey:
				# namekey = namekey.replace("linear", "last_linear")
			new_state_dict[namekey] = v
			# print(namekey, end="|")
		model.load_state_dict(new_state_dict)
		model.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(64),
					# nn.Dropout(0.8),
					nn.Linear(64, config.num_classes),
				)
				
	elif model_name == "resnet200":
		print("train %s"%model_name)
		model = resnet200()#pretrained="imagenet"
		model.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
		model.avg_pool = nn.AdaptiveAvgPool2d(1)
		model.last_linear = nn.Sequential(
					# nn.BatchNorm1d(2048),
					# nn.Dropout(0.8),
					nn.Linear(64, config.num_classes),
				)
	else:
		print("Didn't choose which model to be used!")
		raise ValueError
		
	return model

class MultiModalNet(nn.Module):
	def __init__(self, config, sep_pretrained=True, visit_channels = 7, fold = 0):
		super().__init__()
		
		name = config.model_name.split("_")
		name[1] = name[1].split("-")[0]
		
		self.image_model = get_net(name[0], config, img_channels = 3)
		self.visit_model = get_net(name[1], config, img_channels = visit_channels)
		
		v1_seq = nn.Sequential(
					# nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.5),
					# nn.Linear(in_features=2048, out_features=1024, bias=True),
					# nn.ReLU(inplace=True),
					# nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# nn.Dropout(0.5),
					# nn.Linear(1024, 128, bias=True)
					)
		
		features_num = {"resnet50" : 256,
						"resnet56" : 64,
						"seresnext5032x4d" : 128,
						"resnet110" : 64,
						}
		
		if sep_pretrained:
			# configimgname = "seresnext5032x4d-CElog-03washing-lr28-v0" if name[0] == "seresnext5032x4d" else "resnet50-celog"
			configimgname = "seresnext5032x4d-img224-FLlog-03washing-lr28-v0-final-batch32"#"seresnext5032x4d-FLlog-03washing-lr28-v0-final-batch256"
			if "resnet110" in name[1]:
				# configvisitname = "resnet110-celog-visit-pretrained" if visit_channels == 7 else "resnet110-celog-182visit-pretrained"#"resnet110-FLlog-03washing-lr28-v0-final"
				configvisitname = "resnet110-CElog-03washing-lr28-v0-final-nonorm" if visit_channels == 7 else "resnet110-celog-182visit-pretrained"
				
			elif "seresnext5032x4d" in name[1]:
				configvisitname = "seresnext5032x4d-FLlog-03washing-lr28-v0-182logvp"
			else:
				print("Model name error!")
			print("Single model pretrained respectively")
			
			checkpoint_path = "%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, configimgname, str(fold))
			# print(checkpoint_path)
			if os.path.isfile(checkpoint_path):
				print("=> loading checkpoint '{}'".format(checkpoint_path))
				best_model = torch.load(checkpoint_path)#["state_dict"]
				#best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
				# self.image_model.load_state_dict(best_model)
				new_state_dict = OrderedDict()
				for k, v in best_model["state_dict"].items():
					namekey = k[7:] if "module." in k else k# remove `module.`
					# if "linear" in namekey:
						# namekey = namekey.replace("linear", "last_linear")
					new_state_dict[namekey] = v
					# print(namekey, end="|")
				self.image_model.load_state_dict(new_state_dict)
				
			# checkpoint_path = "%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, configvisitname, str(fold))
			# if os.path.isfile(checkpoint_path):
				# print("=> loading checkpoint '{}'".format(checkpoint_path))
				# best_model = torch.load(checkpoint_path)["state_dict"]
				# new_state_dict = OrderedDict()
				# for k, v in best_model.items():
					# namekey = k[7:] if "module." in k else k# remove `module.`
					# # if "linear" in namekey:
						# # namekey = namekey.replace("linear", "last_linear")
					# new_state_dict[namekey] = v
					# # print(namekey, end="|")
				# if "last_linear.1.weight" in new_state_dict.keys():
					# print("Replace lastlinear parameters name")
					# new_state_dict["last_linear.0.weight"] = new_state_dict.pop("last_linear.1.weight")
					# new_state_dict["last_linear.0.bias"] = new_state_dict.pop("last_linear.1.bias")
				# #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
				# self.visit_model.load_state_dict(new_state_dict)
		
		img_features = features_num[name[0]]
		visit_features = features_num[name[1]]
		
		self.image_model.last_linear = nn.Sequential(nn.Dropout(0),
													#nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
													nn.Linear(in_features=2048, out_features=img_features, bias=True))#
		# self.image_model.last_linear = nn.Linear(in_features=2048, out_features=img_features, bias=True)
		self.visit_model.last_linear = nn.Dropout(0)#nn.Sequential(*list(self.visit_model.children())[:-1])
		
		if "seresnext5032x4d" in name[1]:
			self.visit_model.last_linear = nn.Sequential(#nn.Dropout(0.2),
													nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
													nn.Linear(in_features=2048, out_features=visit_features, bias=True))#
		
		self.cls = nn.Sequential(nn.Dropout(0.05), 
								# nn.BatchNorm1d(visit_features+img_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								  nn.Linear(visit_features+img_features, config.num_classes))#nn.Dropout(0.05), 
		
		# else:
			# print("No single model pretrained respectively")
			# self.visit_model.last_linear = nn.Dropout(0)
			# if name[0] == "resnet110":
				# self.image_model.last_linear = nn.Dropout(0)
				# self.cls = nn.Linear(128, config.num_classes)
				
			# elif name[0] == "seresnext5032x4d":

				# self.image_model.last_linear = nn.Linear(in_features=2048, out_features=128, bias=True)

				# self.cls = nn.Sequential(
						# nn.ReLU(inplace=True),
						# nn.BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
						# nn.Dropout(0.5),
						# nn.Linear(192, config.num_classes)
					# )
				
			# else:

				# self.image_model.last_linear = nn.Sequential(
													# nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
													# nn.Dropout(0.5),
													# nn.Linear(in_features=2048, out_features=1024, bias=True),
													# nn.ReLU(inplace=True),
													# nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
													# nn.Dropout(0.5),
													# nn.Linear(1024, 256, bias=True)
					# )
				# self.cls = nn.Sequential(
						# nn.ReLU(inplace=True),
						# nn.BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
						# nn.Dropout(0.5),
						# nn.Linear(320, config.num_classes)
					# )
	
	def features(self, image_data = None, visit_data = None):
		x_img = self.image_model(image_data)
		x_vis = self.visit_model(visit_data)
		
		x_cat = torch.cat((x_img,x_vis),1)
		return x_cat
	
	def forward(self, image_data = None, visit_data = None):
		x_cat = self.features(image_data, visit_data)
		x_cat = self.cls(x_cat)
		return x_cat
	
	def modify_model(self):
		# pass
		print("Modifing model...")
		image_ll = [list(self.image_model.last_linear.children())[-1]]#nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)] + 
		self.image_model.last_linear = nn.Sequential(*image_ll)
		self.cls = nn.Sequential(*([nn.Dropout(0.05)] + [list(self.cls.children())[-1]]))

class DualVisitNet(nn.Module):
	def __init__(self, config, sep_pretrained=True, visit_channels = 7, fold = 0):
		super().__init__()
		
		name = config.model_name.split("_")
		name[1] = name[1].split("-")[0]
		
		self.vis7_model = get_net(name[0], config, img_channels = 3)
		self.visit_model = get_net(name[1], config, img_channels = visit_channels)
		
		features_num = {"resnet50" : 256,
						"resnet56" : 64,
						"seresnext5032x4d" : 128,
						"resnet110" : 64,
						}
		
		# if sep_pretrained:
			# # configimgname = "seresnext5032x4d-CElog-03washing-lr28-v0" if name[0] == "seresnext5032x4d" else "resnet50-celog"
			# configimgname = "seresnext5032x4d-img224-FLlog-03washing-lr28-v0-final-batch32"#"seresnext5032x4d-FLlog-03washing-lr28-v0-final-batch256"
			# if "resnet110" in name[1]:
				# # configvisitname = "resnet110-celog-visit-pretrained" if visit_channels == 7 else "resnet110-celog-182visit-pretrained"#"resnet110-FLlog-03washing-lr28-v0-final"
				# configvisitname = "resnet110-CElog-03washing-lr28-v0-final-nonorm" if visit_channels == 7 else "resnet110-celog-182visit-pretrained"
				
			# elif "seresnext5032x4d" in name[1]:
				# configvisitname = "seresnext5032x4d-FLlog-03washing-lr28-v0-182logvp"
			# else:
				# print("Model name error!")
			# print("Single model pretrained respectively")

			# checkpoint_path = "%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, configvisitname, str(fold))
			# if os.path.isfile(checkpoint_path):
				# print("=> loading checkpoint '{}'".format(checkpoint_path))
				# best_model = torch.load(checkpoint_path)["state_dict"]
				# new_state_dict = OrderedDict()
				# for k, v in best_model.items():
					# namekey = k[7:] if "module." in k else k# remove `module.`
					# # if "linear" in namekey:
						# # namekey = namekey.replace("linear", "last_linear")
					# new_state_dict[namekey] = v
					# # print(namekey, end="|")
				# if "last_linear.1.weight" in new_state_dict.keys():
					# print("Replace lastlinear parameters name")
					# new_state_dict["last_linear.0.weight"] = new_state_dict.pop("last_linear.1.weight")
					# new_state_dict["last_linear.0.bias"] = new_state_dict.pop("last_linear.1.bias")
				# # best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
				# self.visit_model.load_state_dict(new_state_dict)
		
		vis7_features = features_num[name[0]]
		visit_features = features_num[name[1]]
		
		self.vis7_model.last_linear = nn.Dropout(0)
		self.visit_model.last_linear = nn.Dropout(0)#nn.Sequential(*list(self.visit_model.children())[:-1])
		
		self.cls = nn.Sequential(nn.Dropout(0.1), 
								# nn.BatchNorm1d(visit_features+img_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
								  nn.Linear(visit_features+vis7_features, config.num_classes))#nn.Dropout(0.05), 
	
	def features(self, vis7_data = None, visit_data = None):
		x_vis7 = self.vis7_model(vis7_data)
		x_vis = self.visit_model(visit_data)
		
		x_cat = torch.cat((x_vis7, x_vis),1)
		return x_cat
	
	def forward(self, vis7_data = None, visit_data = None):
		x_cat = self.features(vis7_data, visit_data)
		out = self.cls(x_cat)
		return out
	
	def modify_model(self):
		# pass
		print("Modifing model...")
		image_ll = [list(self.vis7_model.last_linear.children())[-1]]#nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)] + 
		self.vis7_model.last_linear = nn.Sequential(*image_ll)
		self.cls = nn.Sequential(*([nn.Dropout(0.05)] + [list(self.cls.children())[-1]]))
	

# class SiameseNet(nn.Module):
	# def __init__(self, config, drop, pretrained=True, visit_channels = 7, fold = 0):
		# super().__init__()
		
		# name = config.model_name.split("_")
		# # print(name)
		# assert(name[0] == name[1].split("-")[0])
		
		# self.main_model = get_net(name[0], config)
		
		# self.imginput_model = self.main_model.layer0
		# self.visitinput_model = nn.Sequential(
			# nn.Conv2d(visit_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)), 
			# nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			# nn.ReLU(inplace=True),
			# nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
			# )
		
		# self.main_model.layer0 = nn.Dropout(0, inplace=True)
		# out_num = 128 #if name[0] == "seresnext5032x4d" else 128
		# self.main_model.last_linear = nn.Sequential(
			# # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			# # nn.Dropout(0.5),
			# # nn.Linear(in_features=2048, out_features=1024, bias=True),
			# # nn.ReLU(inplace=True),
			# # nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			# nn.Dropout(0.2),
			# nn.Linear(2048, out_num, bias=True),
			# # nn.ReLU(inplace=True),
		# )
		
		# # v1_seq = nn.Sequential(
					# # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# # nn.Dropout(0.5),
					# # nn.Linear(in_features=2048, out_features=1024, bias=True),
					# # nn.ReLU(inplace=True),
					# # nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
					# # nn.Dropout(0.5),
					# # nn.Linear(1024, 128, bias=True)
					# # )

		# self.cls = nn.Sequential(#nn.BatchNorm1d(out_num*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
								# nn.Dropout(0.2), 
								# nn.Linear(out_num*2, config.num_classes))#nn.Dropout(0.05), 
		
		
		# # self.cls = nn.Sequential(
				# # nn.ReLU(inplace=True),
				# # nn.BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				# # nn.Dropout(0.5),
				# # nn.Linear(320, config.num_classes)
			# # )

	# def forward(self, image_data = None, visit_data = None):
		
		# x_img = self.imginput_model(image_data)
		# x_vis = self.visitinput_model(visit_data)
		# x_img = self.main_model(x_img)
		# x_vis = self.main_model(x_vis)
		
		# x_cat = torch.cat((x_img,x_vis),1)
		# x_cat = self.cls(x_cat)
		
		# return x_cat