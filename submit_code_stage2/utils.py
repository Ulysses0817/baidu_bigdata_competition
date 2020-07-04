import os
import sys 
import json
import torch
import shutil
import numpy as np 
# from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable


class Linear(nn.Module):
	def __init__(self, in_features, out_features, dropout=0.0):
		super(Linear, self).__init__()

		self.linear = nn.Linear(in_features=in_features, out_features=out_features)
		if dropout > 0:
			self.dropout = nn.Dropout(p=dropout)
		self.reset_params()

	def reset_params(self):
		nn.init.kaiming_normal_(self.linear.weight)
		nn.init.constant_(self.linear.bias, 0)

	def forward(self, x):
		if hasattr(self, 'dropout'):
			x = self.dropout(x)
		x = self.linear(x)
		return x

def visit2array(strings, date2position, datestr2dateint, str2int, path):
	init = np.zeros((7, 26, 24))
	for string in strings:
		temp = []
		for item in string.split(','):
			temp.append([item[0:8], item[9:].split("|")])
		for date, visit_lst in temp:
			# x - 第几周
			# y - 第几天
			# z - 几点钟
			# value - 到访的总人数
			x, y = date2position[datestr2dateint[date]]
			for visit in visit_lst: # 统计到访的总人数
				init[x][y][str2int[visit]] += 1
	# return init
	path = path.replace("part", "partarray").replace("txt", "npy")
	if not os.path.exists(os.path.dirname(path.strip("//"))):
		print("making dir -- ", path)
		os.makedirs(os.path.dirname(path.strip("//")))
	np.save(path, init)

def cls_wts(label_dict, mu=0.5):
	n_labels = sum(label_dict.values())
	prob_dict, prob_dict_bal = {}, {}
	max_ent_wt = 1/9
	for i in range(9):
		prob_dict[i] = label_dict[i]/n_labels
		if prob_dict[i] > max_ent_wt:
			prob_dict_bal[i] = prob_dict[i]-mu*(prob_dict[i] - max_ent_wt)
		else:
			prob_dict_bal[i] = prob_dict[i]+mu*(max_ent_wt - prob_dict[i])            
	return prob_dict, prob_dict_bal
	
def create_class_weight(labels_dict, mu=0.5):
	total = sum(labels_dict.values())
	keys = labels_dict.keys()
	class_weight = dict()
	class_weight_log = dict()

	for key in keys:
		score = total / float(labels_dict[key])
		score_log = np.log(mu * total / float(labels_dict[key]))
		class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
		class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

	return class_weight, class_weight_log
	
# save best model
def save_checkpoint(state, is_best_loss,is_best_acc,fold, preds_y=None, config=None):
	filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
	# torch.save(state, filename)
	if is_best_loss:
		torch.save(state, "%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
	if is_best_acc:
		torch.save(state, "%s/%s_fold_%s_model_best_acc.pth.tar"%(config.best_models,config.model_name,str(fold)))

def accuracy(y_pred, y_actual, topk=(1, ), np=True):
	"""Computes the precision@k for the specified values of k"""
	if not (isinstance(y_pred, torch.Tensor)&isinstance(y_actual, torch.Tensor)):
		y_pred = torch.Tensor(y_pred)
		y_actual = torch.Tensor(y_actual)
	maxk = max(topk)
	batch_size = y_actual.size(0)
	# print(batch_size)
	_, pred = y_pred.topk(maxk, 1, True, True)
	pred = pred.t()
	# print(pred.size())
	# y_actual = torch.max(y_actual, 1)[1]
	# print(y_actual.size())
	correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	if np:
		return res[0].cpu().data.numpy()
	else:
		return res[0]

#-----multi-gpu training---------
def load_network(network):
	save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
	state_dict = torch.load(save_path)
	# create new OrderedDict that does not contain `module.`
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		namekey = k[7:] # remove `module.`
		new_state_dict[namekey] = v
	# load params
	network.load_state_dict(new_state_dict)
	return network

# evaluate meters
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
# print logger
class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout  #stdout
		self.file = None

	def open(self, file, mode=None):
		if mode is None: mode ='w'
		self.file = open(file, mode)

	def write(self, message, is_terminal=1, is_file=1 ):
		if '\r' in message: is_file=0

		if is_terminal == 1:
			self.terminal.write(message)
			self.terminal.flush()
			#time.sleep(1)

		if is_file == 1:
			self.file.write(message)
			self.file.flush()

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass
	
class AccCELoss(nn.Module):
	def __init__(self):
		super(AccCELoss, self).__init__()
		
	def forward(self, y_pred, y_actual):
		'''Focal loss.
		Args:
		  x: (tensor) sized [N,D].
		  y: (tensor) sized [N,].
		Return:
		  (tensor) focal loss.
		'''
		celoss = F.cross_entropy(y_pred, y_actual)
		acc_loss = 1 - accuracy(y_pred, y_actual, np=False)/100
		
		return torch.add(celoss, acc_loss)

def dice_loss(input, target):
	input = torch.sigmoid(input)
	smooth = 1.

	iflat = input.view(-1)
	tflat = target.view(-1)
	intersection = (iflat * tflat).sum()
	
	return 1 - ((2. * intersection + smooth) /
			  (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
	def __init__(self, gamma = 2):
		super().__init__()
		self.gamma = gamma
		
	def forward(self, input, target):
		# Inspired by the implementation of binary_cross_entropy_with_logits
		target = target.reshape((-1, 1)).cpu()
		target = torch.zeros((target.size()[0], 9)).scatter_(1, target, 1).cuda()
		if not (target.size() == input.size()):
			raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
		
		max_val = (-input).clamp(min=0)
		loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

		# This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
		invprobs = F.logsigmoid(-input * (target * 2 - 1))
		loss = (invprobs * self.gamma).exp() * loss
		
		return loss.mean()
		
def get_learning_rate(optimizer):
	lr=[]
	for param_group in optimizer.param_groups:
	   lr +=[ param_group['lr'] ]

	#assert(len(lr)==1) #we support only one param_group
	lr = lr[0]

	return lr

def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)


	else:
		raise NotImplementedError
