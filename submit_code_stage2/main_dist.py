import os 
import sys
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 

from utils import *
from data import PlanetDataset
from tqdm import tqdm 
from config_dist import config
from datetime import datetime
from models.model import *
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from timeit import default_timer as timer
# from sklearn.metrics import f1_score

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# config.batch_size = 128*1
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


def train(train_loader, model, criterion, optimizer, epoch, valid_loss, best_results, start):
	losses = AverageMeter()
	acc = AverageMeter()
	model.train()
	targets = []
	outputs = []
	for i,(images, visits, target) in enumerate(train_loader):
		images_var = images.cuda(non_blocking=True)
		visits_var = visits.cuda(non_blocking=True)
		# target = target.astype(np.int64)
		target = torch.from_numpy(np.array(target)).long().cuda(non_blocking=True)
		
		# compute output
		# print('compute output',end='',flush=True)
		# print('target size',target.size(),flush=True)

		# try:
		# torch.cuda.empty_cache()
		output = model(images_var, visits_var)
		# except RuntimeError as exception:
			# if "out of memory" in str(exception):
				# print("WARNING: out of memory")
				# if hasattr(torch.cuda, 'empty_cache'):
					# torch.cuda.empty_cache()
			# else:
				# raise exception
		
		targets.append(target.cpu())
		outputs.append(output.cpu())
		loss = criterion(output, target)
		losses.update(loss.item(), images_var.size(0))
		
		accuracy_batch = accuracy(output, target)
		acc.update(accuracy_batch, images_var.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i % 5 == 0:
			# print(stage, optimizer)
			print('\r',end='',flush=True)
			message = '%s %5.1f %6.1f         |         %0.3f  %0.3f          |         %4.3f   %0.4f         |     %8s  %8s        | %s' % (\
					"train", i/len(train_loader) + epoch, epoch,
					losses.avg, acc.avg, 
					valid_loss[0], valid_loss[1], 
					str(best_results[0])[:8],str(best_results[1])[:8],
					time_to_str((timer() - start),'min'))
			log.write(message)# , end='',flush=True)
			log.write("\n", is_terminal=0)
			time.sleep(0.01)
	#log.write(message)
	log.write("\n")
	final_target = torch.cat(targets)
	final_output = torch.cat(outputs)
	final_acc_score = accuracy(final_output, final_target)  
	return [losses.avg, final_acc_score]

# 2. evaluate fuunction
def evaluate(val_loader, model, criterion, epoch, train_loss, best_results, start):
	# only meter loss and f1 score
	losses = AverageMeter()
	acc = AverageMeter()
	# switch mode for evaluation
	model.cuda()
	model.eval()
	targets = []
	outputs = []
	with torch.no_grad():
		for i, (images, visits, target) in enumerate(val_loader):
			images_var = images.cuda(non_blocking=True)
			visits_var = visits.cuda(non_blocking=True)
			target = torch.from_numpy(np.array(target)).long().cuda(non_blocking=True)
			targets.append(target)
			output = model(images_var, visits_var)
			outputs.append(output)
			loss = criterion(output, target)
			losses.update(loss.item(),images_var.size(0))
			if i % 10 == 0:
				print('\r',end='',flush=True)
				message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f          |         %0.3f  %4s        |     %8s  %8s       | %s' % (\
						"val", i/len(val_loader) + epoch, epoch,                    
						train_loss[0], train_loss[1], 
						losses.avg, "None",
						str(best_results[0])[:8],str(best_results[1])[:8],
						time_to_str((timer() - start),'min'))

				log.write(message)#, end='',flush=True
				log.write("\n", is_terminal=0)
				# time.sleep(0.01)
		final_target = torch.cat(targets)
		final_output = torch.cat(outputs)
		# if epoch<10:
		final_acc_score = accuracy(final_output, final_target)  
		#log.write(message)
		print('\r',end='',flush=True)
		message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f          |         %0.3f  %0.4f        |     %8s  %8s       | %s' % (\
				"val", i/len(val_loader) + epoch, epoch,                    
				train_loss[0], train_loss[1], 
				losses.avg, final_acc_score,
				str(best_results[0])[:8],str(best_results[1])[:8],
				time_to_str((timer() - start),'min'))

		log.write(message)#
		log.write("\n", is_terminal=1)
		# time.sleep(0.01)
			
		# else:
			# c_thr = ChooseThre()
			# best_threshold, final_f1_score, _count = c_thr.calcul_th(final_target,final_output.sigmoid().cpu())
			# for i in range(c_thr.len):
				# #print('\r',end='',flush=True)
				# message = '%s  %5.4f %6.5f|%6.5f |         %0.0f  %0.4f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
						# "val",  c_thr.best_thr[i] , c_thr.best_val[i][0], c_thr.best_val[i][1],                    
						# c_thr.count[i], train_loss[1], 
						# losses.avg, final_f1_score,
						# str(best_results[0])[:8],str(best_results[1])[:8],
						# time_to_str((timer() - start),'min'))
				
				# log.write(message)#, end='',flush=True
				# log.write("\n", is_terminal=1)
				# time.sleep(0.01)
				
		# log.write("\n")
		
	return [losses.avg, final_acc_score]

# test model on public dataset and save the probability matrix
def test(test_loader, model, folds, all_train_loader=None):
#     sample_submission_df = pd.read_csv("./dataset/test_visit.csv")
	#3.1 confirm the model converted to cuda
	filenames, labels_te, labels_tr, submissions= [],[],[],[]
	labels_true = []
	model.cuda()
	model.eval()
	submit_results = []
	try:
		with tqdm(test_loader, ascii = True) as t:
			for img_data, visit_data, filepath in t:
				#3.2 change everything to cuda and get only basename
				filepath = [x for x in filepath]
				with torch.no_grad():
					image_var = img_data.cuda(non_blocking=True)
					visit_var = visit_data.cuda(non_blocking=True)
					y_pred = model(image_var, visit_var)
					label = y_pred.sigmoid().cpu().data.numpy()
					#print(label > 0.5)

					submit_results.append(np.argmax(label, 1))
					labels_te.append(np.squeeze(label))

					filenames.append(np.squeeze(filepath))

		if not isinstance(all_train_loader, type(None)):
			with tqdm(all_train_loader, ascii = True) as t:
				for img_data, visit_data, target in t:
					#3.2 change everything to cuda and get only basename
					target = np.squeeze(target).cpu().data.numpy()
					with torch.no_grad():
						image_var = img_data.cuda(non_blocking=True)
						visit_var = visit_data.cuda(non_blocking=True)
						y_pred = model(image_var, visit_var)
						label = y_pred.sigmoid().cpu().data.numpy()
						labels_tr.append(np.hstack([np.squeeze(label), target.reshape(-1, 1)]))
			return np.concatenate(labels_te), np.concatenate(labels_tr)
	except:
		t.close()
		raise
	t.close()

# 	sample_submission_df['Predicted'] = np.concatenate(submit_results)
# 	sample_submission_df['IterId'] = np.concatenate(filenames)
# 	sample_submission_df.to_csv('./submit/%s_bestloss_submission%s.csv'%(config.model_name, folds), index=None)
	return np.concatenate(labels_te), None

# 学习率衰减：lr = lr / lr_decay
def adjust_learning_rate(model, stage=3, lr = None, base_params=None):

	# global lr
	
	lr_decay = [10, 10, 10, 20, 1/2]#8, 8]
	lr = lr / lr_decay[stage-1]
	log.write("Learning rate changed to %s"%lr)
	if stage >= 5:
		log.write("choosing optimizer sgd")
		return optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=config.weight_decay), lr
	else:
		if isinstance(base_params, type(None)):
			return optim.Adam(model.parameters(), lr, weight_decay=config.weight_decay), lr
		else:
			log.write("finetune lr:%s"%(lr/50))
			return optim.Adam([{'params': base_params, 'lr': lr/50},
											{'params': model.module.cls.parameters()},
											{'params': model.module.image_model.last_linear.parameters()}], lr, weight_decay=config.weight_decay)

		
# 4. main function
def main(fold = 0, resume = False, train_data_list = None, val_data_list = None, test_files = None, all_files = None, dtype="mixed"):

	log.open("logs/%s_log_train%s.txt"%(config.model_name, fold), mode="a")
	
	# 4.1 mkdirs
	print("4.1 mkdirs")
	if not os.path.exists(config.submit):
		os.makedirs(config.submit)
	if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
		os.makedirs(config.weights + config.model_name + os.sep +str(fold))
	if not os.path.exists(config.best_models):
		os.mkdir(config.best_models)
	if not os.path.exists("./logs/"):
		os.mkdir("./logs/")
	
	# 4.2 get model
	print("4.2 get model")
	# model = get_net(config.model_name, config)
	vc = 1 if dtype in ["mixed_v1"] else 7
	# vc = 2
	model = MultiModalNet(config, sep_pretrained=True, visit_channels=vc, fold=fold)

	# pre_model_name = {7:"seresnext5032x4d_resnet110-FLlog-03washing-vp-adjustlr-v0pretrained", 1:"seresnext5032x4d_resnet110-celog-03washing-182vp-adjustlr-skf"}
	# print(pre_model_name[vc])
	# best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, pre_model_name[vc], str(fold)))
	# model.load_state_dict(best_model["state_dict"])
	model.modify_model()
	
	# model.cuda()
	lr = config.lr
	print("Initial learning rate:", lr)
	stage = 0
	start_epoch = 0
	best_loss = 999
	best_accuracy = 0
	best_results = [np.inf, 0]
	val_metrics = [np.inf, 0]

	# MultiGpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.device_count() > 1:
		# config.batch_size = config.batch_size * torch.cuda.device_count()
		print("Let's use", torch.cuda.device_count(), "GPUs - ", config.batch_size)
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		model = nn.DataParallel(model)#, device_ids=[0, 2]
	model.to(device)
	
	#==================================#
	#        分层模型参数设置
	# ignored_params = list(map(id, model.module.cls.parameters())) + list(map(id, model.module.image_model.last_linear.parameters())) # 返回的是parameters的 内存地址
	# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters()) 
	#==================================#
	
	# if fold in [0]:#, 1, 2, 3
		# resume = True
		
	# optionally resume from a checkpoint
	if resume:
		print("optionally resume from a checkpoint")
		checkpoint_path = "%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, config.model_name, str(fold))
		if os.path.isfile(checkpoint_path):
			print("=> loading checkpoint '{}'".format(checkpoint_path))
			checkpoint = torch.load(checkpoint_path)
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in checkpoint.items():
				namekey = k[7:] if "module." in k else k# remove `module.`
				new_state_dict[namekey] = v
				print(namekey)
			start_epoch = new_state_dict['epoch']# if fold != 0 else 6
			best_results[0] = new_state_dict['best_loss']
			best_results[1] = new_state_dict['best_accuracy']
			stage = new_state_dict['stage']
			lr = checkpoint["optimizer"]['param_groups'][0]["lr"]
			# lr = checkpoint['lr']
			# lr = 1e-6
			
			# print(checkpoint['state_dict']["conv1_7x7_s2.weight"].shape, "?????????????")
			model.load_state_dict(checkpoint['state_dict'])
			# 如果中断点恰好为转换stage的点，需要特殊处理
			if start_epoch in np.cumsum(config.stage_epochs)[:-1]:
				stage += 1
				
				_best_model = torch.load(checkpoint_path)
				model.load_state_dict(_best_model["state_dict"])
				optimizer, lr = adjust_learning_rate(model, stage, lr, base_params=None)
				
				log.write('Step into next stage')
				log.write("\n")
				# time.sleep(0.01)
			print("=> loaded checkpoint (epoch {})".format(new_state_dict['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(checkpoint_path))
	
	# 4.3 optimizer and criterion set up
	print("4.3 optimizer and criterion set up")
	# optimizer = optim.SGD(model.parameters(),lr = config.lr, momentum=0.9, weight_decay=9e-5)
	# optimizer = optim.Adam([{'params': base_params, 'lr': lr/50},
							# {'params': model.module.cls.parameters()},
							# {'params': model.module.image_model.last_linear.parameters()}], lr, weight_decay=config.weight_decay)
	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=config.weight_decay)
	
	# Weighted CE
	# name_label_dict = dict(all_files["Target"].value_counts())
	# di = cls_wts(name_label_dict)[1]
	# class_weights = [item[1] for item in sorted(di.items(), key=lambda x: x[0])]
	# class_weights = torch.FloatTensor(class_weights).cuda()
	criterion = nn.CrossEntropyLoss().cuda()#nn.BCEWithLogitsLoss().cuda()weight=class_weights
	# criterion = FocalLoss().cuda()

	# 4.4 load dataset
	print("4.4 load dataset")
	# now lets initialize samplers 
	# samples_weight = np.array([class_weights[t] for t in train_data_list.Target])
	# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))#sampler=train_sampler
	
	# print("load train dataset")
	train_gen = PlanetDataset(train_data_list, mode="train", dtype=dtype, config=config, dehaze=False, randomdehaze=False)
	train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=16)
	# print(len(train_loader), len(samples_weight))
	# print("load val dataset")
	val_gen = PlanetDataset(val_data_list, augument=False, mode="train", dtype=dtype, config=config, dehaze=False, randomdehaze=False)
	val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=16)
	# print("load test dataset")
	# all_train_gen = PlanetDataset(all_files, augument=False, mode="train", dtype=dtype, config=config)
	# all_train_loader = DataLoader(all_train_gen, 1024, shuffle=False, pin_memory=True, num_workers=16)
	
	test_gen = PlanetDataset(test_files,  augument=False,  mode="test", dtype=dtype, config=config, dehaze=False, randomdehaze=False)
	test_loader = DataLoader(test_gen, 1024, shuffle=False, pin_memory=True, num_workers=16)

	# scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
	start = timer()

	log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
	log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
	log.write('mode    iter  epoch        |         loss   accuracy        |         loss   accuracy       |         loss   accuracy       | time       |\n')
	log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

	#train
	# print(start_epoch, config.epochs)
	for epoch in range(start_epoch, config.epochs):
		# scheduler.step(epoch)
		# train
		# lr = get_learning_rate(optimizer)
		# print("train")
		# if fold in [0]:
			# break
		# print(stage, optimizer)
		train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
		# print("val")
		val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
		# check results 
		is_best_loss = val_metrics[0] <= best_results[0]
		best_results[0] = min(val_metrics[0],best_results[0])
		is_best_acc = val_metrics[1] >= best_results[1]
		best_results[1] = max(val_metrics[1],best_results[1])
		
		# save model
		save_checkpoint({
					"epoch":epoch + 1,
					"model_name":config.model_name,
					"state_dict":model.state_dict(),
					"best_loss":best_results[0],
					"optimizer":optimizer.state_dict(),
					"fold":int(fold),
					"best_accuracy":best_results[1],
					"stage":stage,
		},is_best_loss,is_best_acc,fold,config=config)
		# print logs
		print('\r',end='',flush=True)
		log.write('%s   %5.1f     %6.1f    |         %5.4f %5.4f         |         %5.4f %5.4f        |       %8s  %8s     | %s' % (\
				"best", epoch, epoch,                    
				train_metrics[0], train_metrics[1], 
				val_metrics[0], val_metrics[1],
				str(best_results[0])[:8],str(best_results[1])[:8],
				time_to_str((timer() - start),'min'))
			)
		log.write("\n")
		# time.sleep(0.01)
		
		# if epoch == 1 and fold == 0:
			# print("test%s test%s"%(epoch, fold))
			# test(test_loader, model, fold, all_train_loader)
		# 判断是否进行下一个stage

		if (epoch + 1) in np.cumsum(config.stage_epochs)[:-1]:

			stage += 1
			
			_best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
			model.load_state_dict(_best_model["state_dict"])
			optimizer, lr = adjust_learning_rate(model, stage, lr, base_params=None)
			
			log.write('Step into next stage')
			log.write("\n")
			time.sleep(0.01)
	
	best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models, config.model_name,str(fold)))
	#best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
	model.load_state_dict(best_model["state_dict"])
	return test(test_loader, model, fold, all_train_loader=None)
	
# def transfer(fold = 0):

	# model = get_net()
	# model.cuda()
	
	# # load dataset
	# test_files = pd.read_csv("./mfs/human/sample_submission.csv")
	# test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
	# test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=16)
	
	# best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
	# #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
	# model.load_state_dict(best_model["state_dict"])
	# test(test_loader,model,fold)

# Solve MultiGpu Error
class Net(nn.Module):
	def __init__(self, model):
		super(Net, self).__init__()
		self.l1 = nn.Sequential(*list(model.children())[:-1]).to('cuda:0')
		self.last = list(model.children())[-1]

	def forward(self, x):
		x = self.l1(x)
		x = x.view(x.size()[0], -1)
		x = self.last(x)
		return x
		
if __name__ == "__main__":
	if not os.path.exists("./logs/"):
		os.mkdir("./logs/")
	
	log = Logger()
	
	all_files = pd.read_csv("./dataset/final_train.csv")
	all_files.Id_y = all_files.Id_v182.apply(lambda x: x.replace("visit", "visit182"))
	all_files = all_files.loc[(all_files.black_ratio<0.3)]#|(all_files.same_pixel>70)
	#print(all_files)
	test_files = pd.read_csv("./dataset/final_test1.csv")
	# test_files.Id_x = test_files.Id_x.apply(lambda x: "."+x)
	test_files.Id_y = test_files.Id_y.apply(lambda x: x.replace("visit", "visit182"))
	# test_files = test_files.loc[(test_files.black_ratio<0.3)|(test_files.same_pixel>70)]
	# train_data_list,val_data_list = train_test_split(all_files, test_size = 0.2, random_state = 2050)
	# val_data_list.to_csv("val_data%s.csv"%fold, index=False)
	
	## data_old---->train
	all_files_pre = pd.read_csv("./dataset/pre_train.csv")
	all_files_pre.Id_y = all_files_pre.Id_y.apply(lambda x: x.replace("visit", "visit182"))
	all_files_pre = all_files_pre.loc[(all_files_pre.black_ratio<0.3)]#&(all_files_pre.Target!=0)
	
	k = 5
	kf = KFold(n_splits=k, shuffle=True, random_state=2050)
	
	train_index_pre, test_index_pre = [], []
	for fold, (train_index, test_index) in enumerate(kf.split(all_files_pre)):
		train_index_pre.append(train_index)
		test_index_pre.append(test_index)
		
	#print(parameters)
	all_pred = []
	all_pred_tr = []
	for fold, (train_index, test_index) in enumerate(kf.split(all_files, all_files.Target)):
		X_tr, X_val= all_files.iloc[train_index,:].copy(), all_files.iloc[test_index,:].copy()
		X_tr.to_csv("./temp/final_X_tr%s.csv"%fold, index=False)
		X_val.to_csv("./temp/final_X_val%s.csv"%fold, index=False)
		X_tr_pre, X_val_pre= all_files_pre.iloc[train_index_pre[fold],:].copy(), all_files_pre.iloc[test_index_pre[fold],:].copy()
		X_tr = pd.concat([X_tr, X_tr_pre])
		print( "\nFold ", fold)
		print( "X_tr's size ", X_tr.shape)
		# name_label_dict = dict(X_tr["Target"].value_counts())
		# di = create_class_weight(name_label_dict)[1]
		# class_weights = [item[1] for item in sorted(di.items(), key=lambda x: x[0])]
		# print("class_weights:", class_weights)
		resume=False
		# if fold in [0, 1, 2]:
			# continue
		labels_fold, labels_tr_fold = main(fold, resume, X_tr, X_val, test_files, all_files, dtype="mixed_v1")
		all_pred.append(labels_fold)
		all_pred_tr.append(labels_tr_fold)
		# np.save("./results/0722_mixed_CE_segpre_addcsno0_seresnext5032x4d_182v110_5fold%s_results.npy"%fold, all_pred)
		torch.cuda.empty_cache()
		time.sleep(3)
		# np.save("./results/mixed_pre_182logvpseresnext5032x4d_ttv0-110_5fold_tr_results0702_adjlrinit5.npy", all_pred_tr)