from __future__ import print_function
import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
from Nadam import Nadam
from utils import *
from multimodal import MultiModalDataset,CosineAnnealingLR,MultiModalNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"#,
from tqdm import tqdm 
from config import config
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score
import torch.nn.functional as F
import numpy as np
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')


def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()

    model.train()
    for i,(images,(visit, ),target) in enumerate(train_loader):
        # print(np.array(magic).shape)
        visit=visit.to(device)
        # magic=magic.to(device)
        images = images.to(device)
        indx_target=target.clone()
        target = torch.from_numpy(np.array(target)).float().to(device)
        # compute output
        output = model(images,visit)
        # loss = criterion(output,target)
        print(output.dtype, target.dtype)
        loss = F.binary_cross_entropy_with_logits(output,target,reduction='none')
        loss = loss.sum(1)
        loss, _ = loss.topk(k=int(loss.size(0) * 0.9))
        loss = loss.mean()
        losses.update(loss.item(),images.size(0))
        # f1_batch = f1_score(np.argmax(target.cpu().data.numpy()),np.argmax(output.cpu().data.numpy(),axis=1),average='macro')
        acc_score=accuracy_score(np.argmax(target.cpu().data.numpy(),axis=1),np.argmax(output.cpu().data.numpy(),axis=1))
        # f1.update(f1_batch,images.size(0))
        acc.update(acc_score,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                valid_metrics[0], valid_metrics[1],valid_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [acc.avg,losses.avg,f1.avg]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    acc= AverageMeter()
    # switch mode for evaluation
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images,(visit,),target) in enumerate(val_loader):
            images_var = images.to(device)
            # magic = magic.to(device)
            visit=visit.to(device)
            indx_target=target.clone()
            target = torch.from_numpy(np.array(target)).float().to(device)
            
            output = model(images_var,visit)
            # loss = criterion(output, target)
            loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
            loss = loss.sum(1)
            loss, _ = loss.topk(k=int(loss.size(0) * 0.9))
            loss = loss.mean()
            losses.update(loss.item(),images_var.size(0))
            # f1_batch = f1_score(np.argmax(target.cpu().data.numpy()),np.argmax(output.cpu().data.numpy(),axis=1),average='macro')
            acc_score=accuracy_score(np.argmax(target.cpu().data.numpy(),axis=1),np.argmax(output.cpu().data.numpy(),axis=1))
            # f1.update(f1_batch,images.size(0))
            acc.update(acc_score,images.size(0))
        print('\r',end='',flush=True)
        message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (\
                "val", i/len(val_loader) + epoch, epoch,
                acc.avg,losses.avg,f1.avg,
                train_metrics[0], train_metrics[1],train_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))

        print(message, end='',flush=True)
        log.write("\n")
        #log.write(message)
        #log.write("\n")
        
    return [acc.avg,losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,models):
    N = len(models)
    sample_submission_df = pd.read_csv("../data/test.csv")
    if config.debug:
        sample_submission_df = sample_submission_df.iloc[:100]
    del sample_submission_df['Target']
    sample_submission_df.columns = ['AreaID']
    sample_submission_df['AreaID'] = sample_submission_df['AreaID'].apply(lambda x: str(x).zfill(6))
    #3.1 confirm the model converted to cuda
    result = 0
    stack = {}
    for fold, model in tqdm(enumerate(models)):
        print('fold:', fold)
        filenames, submissions = [], []
        submit_results = []
        labels = pd.DataFrame()
        for i,(input,(visit,),filepath) in tqdm(enumerate(test_loader)):
            #3.2 change everything to cuda and get only basename
            filepath = [os.path.basename(x) for x in filepath]
            with torch.no_grad():
                image_var = [x.to(device) for x in input]
                # magic = magic.to(device)
                visit=visit.to(device)
                y_pred = np.array([F.softmax(model(test_x, visit)).cpu().data.numpy() for test_x in image_var])
                # print(y_pred.shape)
                y_pred = np.mean(y_pred, axis=0)
                # print(y_pred.shape)
                label=y_pred
                # labels.append(label==np.max(label))
                # labels.append(label)
                # filenames.append(filepath)
            temp = pd.DataFrame({'file':filepath})
            for i in range(9):
                temp['P_'+str(i)] = label[:,i]
            labels = pd.concat((labels, temp))
        col = ['P_'+str(i) for i in range(9)]
        labels = np.array(labels[col])
        print(labels.shape)
        stack[str(fold)] = labels
        # labels = np.array(labels).squeeze()
        result += labels
        print(result.shape)
        # for row in np.concatenate(labels):
        #     subrow=np.argmax(row)
        #     submissions.append(subrow)
    result /= N
    result = result.argmax(axis=1)
    sample_submission_df['CategoryID'] = [str(x+1).zfill(3) for x in result]
    sample_submission_df.sort_values(by='AreaID', inplace=True)
    sample_submission_df[['AreaID','CategoryID']].to_csv('./submit/%s_bestacc_submission_tta.csv'%config.model_name, sep="\t", header=False, index=None)
    print('len stack:', len(stack))
    with open("../data/stack_TTA.pkl", 'wb') as f:
            pickle.dump(stack, f)


# 4. main function
def main():
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    for fold in range(config.FOLD):
        if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
            os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    # with open('../data/train_lgb.pkl', 'rb') as f:
    #     magic_trains = pickle.load(f)
    # with open('../data/test_lgb.pkl', 'rb') as f:
    #     magic_tests = pickle.load(f)
    # resume = False
    # if resume:
    #     checkpoint = torch.load(r'./checkpoints/best_models/seresnext101_dpn92_defrog_multimodal_fold_0_model_best_loss.pth.tar')
    #     best_acc = checkpoint['best_acc']
    #     best_loss = checkpoint['best_loss']
    #     best_f1 = checkpoint['best_f1']
    #     start_epoch = checkpoint['epoch']

    start = timer()
    # from torchsummary import summary
    # print(summary(model, [(3, 100, 100), (7*26, 24)]))
    all_files = pd.read_csv("../data/train.csv")
    all_files = all_files.sample(frac=1, random_state=666)
    test_files = pd.read_csv("../data/test.csv")
    max_epoch = config.epochs
    if config.debug:
        all_files = all_files.iloc[:1000]
        test_files = test_files.iloc[:100]
        config.batch_size = 2
        max_epoch = 1
    train_label = np.array(all_files['Target'])
    if config.OOF:
        result = np.zeros((len(all_files), 9))
        # print(result.shape)
        skf = StratifiedKFold(n_splits=config.FOLD, random_state=2019, shuffle=False)
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, train_label)):
            print('fold:', fold)
            val_data_list = all_files.iloc[val_idx]

            # load dataset
            val_gen = MultiModalDataset(val_data_list, config.train_data, config.train_vis, augument=False,
                                        mode="train")
            val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True,
                                    num_workers=1)

            best_model = torch.load(
                "%s/%s_fold_%s_model_best_acc.pth.tar" % (config.best_models, config.model_name, str(fold)))
            model = MultiModalNet(drop=0.5)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)
            model.eval()
            model.load_state_dict(best_model["state_dict"])
            result_oof = []
            with torch.no_grad():
                for i, (images, (visit,), target) in tqdm(enumerate(val_loader)):

                    image_var = images.to(device)
                    # print(image_var.shape)
                    # magic = magic.to(device)
                    visit = visit.to(device)
                    indx_target = target.clone()
                    target = torch.from_numpy(np.array(target)).float().to(device)
                    y_oof = np.array(F.softmax(model(image_var, visit)).cpu().data.numpy())
                    # print(y_oof.shape)
                    result_oof.extend(y_oof)
            result_oof = np.array(result_oof)
            print(len(val_idx), result_oof.shape)
            result[val_idx] = result_oof
        print(result.shape)
        with open("../data/oof2.pkl", 'wb') as f:
            pickle.dump(result, f)


    if config.train and config.FOLD > 1:
        # train_data_list,val_data_list = train_test_split(all_files, test_size=0.1, random_state = 2050)
        skf = StratifiedKFold(n_splits=config.FOLD, random_state=2019, shuffle=False)
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, train_label)):
            print('fold:', fold)
            train_data_list = all_files.iloc[train_idx]
            val_data_list = all_files.iloc[val_idx]
            # train_magic = magic_trains.iloc[train_idx]
            # val_magic = magic_trains.iloc[val_idx]
            # load dataset
            train_gen = MultiModalDataset(train_data_list,config.train_data,config.train_vis,mode="train")
            train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=1) #num_worker is limited by shared memory in Docker!

            val_gen = MultiModalDataset(val_data_list,config.train_data,config.train_vis,augument=False,mode="train")
            val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=1)

            start_epoch = 0
            best_acc = 0
            best_loss = np.inf
            best_f1 = 0
            best_results = [0, np.inf, 0]
            val_metrics = [0, np.inf, 0]
            #model
            # 4.2 get model
            model = MultiModalNet(drop=0.5)
            if fold == 0:
                total_num = sum(p.numel() for p in model.parameters())
                trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print('Total', total_num, 'Trainable', trainable_num)
            # 4.3 optim & criterion
            optimizer = Nadam(model.parameters(), lr=5e-4)
            #torch.optim.Adamax(model.parameters(), 0.001)
            # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss().to(device)
            # scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.25)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [6, 12, 18], gamma=0.5)
            # lr_scheduler.ReduceLROnPlateau(optimizer)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12], gamma=0.25)
            # n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
            # scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*2)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)

            #train
            best_acc_epoch = 0
            for epoch in range(0,max_epoch):
                if epoch - best_acc_epoch > 5:
                    break
                scheduler.step(epoch)
                # train
                # train_metrics = None
                train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
                # val
                val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
                # check results
                is_best_acc=val_metrics[0] > best_results[0]
                if is_best_acc:
                    best_acc_epoch = epoch
                best_results[0] = max(val_metrics[0],best_results[0])
                is_best_loss = val_metrics[1] < best_results[1]
                best_results[1] = min(val_metrics[1],best_results[1])
                is_best_f1 = val_metrics[2] > best_results[2]
                best_results[2] = max(val_metrics[2],best_results[2])
                # save model
                save_checkpoint({
                            "epoch":epoch + 1,
                            "model_name":config.model_name,
                            "state_dict":model.state_dict(),
                            "best_acc":best_results[0],
                            "best_loss":best_results[1],
                            "optimizer":optimizer.state_dict(),
                            "fold":fold,
                            "best_f1":best_results[2],
                },is_best_acc,is_best_loss,is_best_f1,fold)
                # print logs
                print('\r',end='',flush=True)
                log.write('%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % (\
                        "best", epoch, epoch,
                        train_metrics[0], train_metrics[1],train_metrics[2],
                        val_metrics[0],val_metrics[1],val_metrics[2],
                        str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                        time_to_str((timer() - start),'min'))
                    )
                log.write("\n")
                time.sleep(0.01)
    if config.train and config.FOLD == 1:
        train_data_list,val_data_list,train_magic,val_magic = train_test_split(all_files,magic_trains, test_size=0.1, random_state = 2050)
        # skf = StratifiedKFold(n_splits=config.FOLD, random_state=2019, shuffle=False)
        # for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, train_label)):
        #     print('fold:', fold)
        #     train_data_list = all_files.iloc[train_idx]
        #     val_data_list = all_files.iloc[val_idx]
        # load dataset
        train_gen = MultiModalDataset(train_data_list,train_magic,config.train_data,config.train_vis,mode="train")
        train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=1) #num_worker is limited by shared memory in Docker!

        val_gen = MultiModalDataset(val_data_list,val_magic,config.train_data,config.train_vis,augument=False,mode="train")
        val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=1)

        start_epoch = 0
        best_acc = 0
        best_loss = np.inf
        best_f1 = 0
        best_results = [0, np.inf, 0]
        val_metrics = [0, np.inf, 0]
        #model
        # 4.2 get model
        model = MultiModalNet(drop=0.5)
        # 4.3 optim & criterion
        optimizer = torch.optim.Adamax(model.parameters(), 0.001)
        # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().to(device)
        # scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.25)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12], gamma=0.25)
        # n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
        # scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*2)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        #train
        best_acc_epoch = 0
        for epoch in range(0,max_epoch):
            if epoch - best_acc_epoch > 5:
                break
            scheduler.step(epoch)
            # train
            # train_metrics = None
            train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
            # val
            val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
            # check results
            is_best_acc=val_metrics[0] > best_results[0]
            if is_best_acc:
                best_acc_epoch = epoch
            best_results[0] = max(val_metrics[0],best_results[0])
            is_best_loss = val_metrics[1] < best_results[1]
            best_results[1] = min(val_metrics[1],best_results[1])
            is_best_f1 = val_metrics[2] > best_results[2]
            best_results[2] = max(val_metrics[2],best_results[2])
            # save model
            save_checkpoint({
                        "epoch":epoch + 1,
                        "model_name":config.model_name,
                        "state_dict":model.state_dict(),
                        "best_acc":best_results[0],
                        "best_loss":best_results[1],
                        "optimizer":optimizer.state_dict(),
                        "fold":fold,
                        "best_f1":best_results[2],
            },is_best_acc,is_best_loss,is_best_f1,fold)
            # print logs
            print('\r',end='',flush=True)
            log.write('%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % (\
                    "best", epoch, epoch,
                    train_metrics[0], train_metrics[1],train_metrics[2],
                    val_metrics[0],val_metrics[1],val_metrics[2],
                    str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                    time_to_str((timer() - start),'min'))
                )
            log.write("\n")
            time.sleep(0.01)
    if config.predict:
        # test data
        models = []
        for fold in range(5):
            best_model = torch.load("%s/%s_fold_%s_model_best_acc.pth.tar"%(config.best_models,config.model_name,str(fold)))
            model = MultiModalNet(drop=0.5)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)
            model.eval()
            model.load_state_dict(best_model["state_dict"])
            models.append(model)
        test_gen = MultiModalDataset(test_files,config.test_data,config.test_vis,augument=False,mode="test",TTA=True)
        test_loader = DataLoader(test_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=1)
        # predict
        test(test_loader,models)
if __name__ == "__main__":
    main()
