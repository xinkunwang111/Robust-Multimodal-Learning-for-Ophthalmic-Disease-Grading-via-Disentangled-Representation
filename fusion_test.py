import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(grandparent_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './dataloader')))



from MMD import MK_MMD
from MMD import compute_js_divergence
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from baseline_models import Res2Net2D,ResNet3D,Multi_ResNet,Multi_EF_ResNet,Multi_CBAM_ResNet,Multi_dropout_ResNet
#from fusion_net import IMDR
from sklearn.model_selection import KFold
from metrics import cal_ece
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from metrics2 import calc_aurc_eaurc,calc_nll_brier
import torch.nn.functional as F
import logging
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
import torch.nn as nn
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')


    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)


    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

def loss_plot(args,loss):
    num = args.end_epochs
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.model_name)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.end_epochs)+'_loss.jpg'
    list_loss = list(loss)
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

import csv

import csv

import os
import csv

def save_results(filename, epoch, loss_meter, acc, precision, recall, f1, auc, specificity=None):
    # 检查文件是否存在
    if not os.path.exists(filename + ".csv"):
        # 如果文件不存在，创建文件并写入表头
        with open(filename + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                'Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity'
            ])

    # 文件存在时，直接追加内容
    with open(filename + ".csv", 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 写入数据行
        row = [
            epoch,
            f"{loss_meter:.6f}",
            f"{acc:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{auc:.4f}"
        ]
        
        # 如果传入了 specificity，添加到行数据中
        if specificity is not None:
            row.append(f"{specificity:.4f}")
        
        writer.writerow(row)




def metrics_plot(arg,name,*args):
    num = arg.end_epochs
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.model_name) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(arg.end_epochs) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)

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

def find_in_u(list_acc,in_list,u_list,class_num=0):
    for i in range(len(list_acc)):
        if list_acc[i] == class_num:
            in_list.append(i)
    in_u_list = np.zeros(len(in_list))
    for j in range(len(in_list)):
        in_u_list[j] = (u_list[in_list[j]])
    return in_u_list



def train(epoch,train_loader,model, best_acc=0.0):
    correct_num, data_num = 0, 0
    model.train()
    loss_meter = AverageMeter()

    all_targets = []
    all_predictions = []
    all_probabilities = []
    num_classes =2

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        #print(f"the data loader {data}")
        data1 = data[0]
        data2 = data[1]

        for v_num in range(len(data1)):
            data1[v_num] = Variable(data1[v_num].float().cuda())
            data2[v_num] = Variable(data2[v_num].float().cuda())
        target = Variable(target.long().cuda())
        data_num += target.size(0)
        # target = Variable(np.array(target)).cuda())

        # refresh the optimizer
        optimizer.zero_grad()

        pred, loss, combined_features1 = model(data1, target, epoch)
        #print(f"the shape of pred is {pred.shape}") (32, number_of_class + 2 )
        
        _, _, combined_features2 = model(data2, target, epoch)
        #print(f"Combined features shape: {combined_features1.shape}") # (32, dim)
        #print(f"poe_embed1 shape: {poe_embed1.shape}") # (32, 2,256)
        #print(f"Combined features shape: {combined_features1.shape}") # (32, dim)
        loss_MDD = MK_MMD(combined_features1, combined_features2)
        #loss_MDD = MK_MMD(poe_embed1, poe_embed2)
        
        # 获取经过 softmax 转换后的概率分布

        # soft_labels1 = F.softmax(poe_embed_cls1, dim=1)
        # soft_labels2 = F.softmax(poe_embed_cls2, dim=1)

        # # 计算 Logits Distillation Loss (Llogits)
        # loss_logit_distillation = compute_js_divergence(soft_labels1, soft_labels2)

        
        #loss = loss.mean()
        #print(f"Loss: {loss.item()} \tLoss MDD: {loss_MDD} \t")
        loss = loss +  loss_MDD 
        predicted = pred.argmax(dim=-1)
        correct_num += (predicted == target).sum().item()

        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        probabilities = torch.nn.functional.softmax(pred, dim=1)
        all_probabilities.extend(probabilities.detach().cpu().numpy())  #

        # compute gradients and take step
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

    aver_acc = correct_num / data_num

    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)



    if len(set(all_targets)) == 2:
        #print("All probabilities:", all_probabilities)
        print("Is NaN in all_probabilities[:, 1]?", np.isnan(all_probabilities[:, 1]).any())

        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        # For multi-class classification
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')


    print("All targets distribution:", np.bincount(all_targets))
    print("All predictions distribution:", np.bincount(all_predictions))

    conf_matrix = confusion_matrix(all_targets, all_predictions)

    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0


    print(f'Train Epoch: {epoch} \tLoss: {loss_meter.avg:.6f} \tAccuracy: {aver_acc:.4f}')
    #print(f"Loss MMD: {loss_MDD:.6f} \tLoss Logit Distillation: {loss_logit_distillation:.6f}")
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f}')


    save_results('log/train_log/'  + args.dataset +  "_"+str(args.Condition_G_Variance)+  "_" +args.name, epoch, loss_MDD, aver_acc, precision, recall, f1, auc,specificity)

    return loss_meter

def val(current_epoch, val_loader, model, best_acc):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    # best_acc = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        data = data[0]
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            pred, loss, _ = model(data, target, current_epoch)
            predicted = pred.argmax(dim=-1)
            loss = loss.mean()


            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            probabilities = torch.nn.functional.softmax(pred, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())

    aver_acc = correct_num / data_num

    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)

    conf_matrix = confusion_matrix(all_targets, all_predictions)

    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0


    print("All targets distribution:", np.bincount(all_targets))
    print("All predictions distribution:", np.bincount(all_predictions))

    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')

    print(f'Validation Epoch: {current_epoch} \tLoss: {loss_meter.avg:.6f} \tAccuracy: {aver_acc:.4f}')
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')

    save_results('log/val_log/'  + args.dataset +  "_"+str(args.Condition_G_Variance)+  "_" +args.name, current_epoch, loss_meter.avg, aver_acc, precision, recall, f1, auc,specificity)
    save_dir = "checkpoint/" + args.dataset  +   "_"+str(args.Condition_G_Variance)  +  "_" + args.name
    if aver_acc > best_acc:
        best_acc = aver_acc
        print('===========> Save best model!')
    
        file_name = os.path.join(save_dir, f"{args.model_name}_{args.dataset}_{args.folder}_best_epoch_{epoch}__{aver_acc}.pth")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, file_name)

    return loss_meter.avg, best_acc

def test(current_epoch, test_loader, model,checkpoint):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data = data[0]
        for v_num in range(len(data)):
            
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            pred, loss ,_ = model(data, target,current_epoch)
            predicted = pred.argmax(dim=-1)
            loss = loss.mean()
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            probabilities = torch.nn.functional.softmax(pred, dim=1)
            all_probabilities.extend(probabilities.detach().cpu().numpy())

    aver_acc = correct_num / data_num

    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    all_probabilities = np.array(all_probabilities)

    # Calculate AUC for binary classification
    if len(set(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probabilities[:, 1])
    else:
        # For multi-class classification
        all_targets_one_hot = label_binarize(all_targets, classes=[0, 1, 2])
        auc = roc_auc_score(all_targets_one_hot, all_probabilities, multi_class='ovr')

    print('====> acc: {:.4f}'.format(aver_acc))
    print(f'Precision: {precision:.4f} \tRecall: {recall:.4f} \tF1 Score: {f1:.4f} \tAUC: {auc:.4f}')
    

    return loss_meter.avg, best_acc





def test_ensemble(args, test_loader,models,epoch):
    if args.dataset == 'MGamma':
        deepen_times = 4
    else:
        deepen_times = 5

    # load ensemble models
    load_model=[]
    # load_model[0]=.23
    for i in range(deepen_times):
        print(i+1)
        if args.num_classes == 2:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
        else:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder +  '_best_epoch.pth')

        load_model.append(torch.load(load_file))
        # KK =model[i]
        models[i].load_state_dict(load_model[i]['state_dict'])
    print('Successfully load all ensemble models')
    for model in models:
        model.eval()
    list_acc = []
    u_list =[]
    in_list = []
    label_list = []
    ece_list=[]
    prediction_list = []
    probability_list = []
    one_hot_label_list = []
    one_hot_probability_list = []
    correct_list=[]
    correct_num, data_num = 0, 0
    epoch_auc = 0
    start_time = time.time()
    time_list= []
    nll_list= []
    brier_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        pred = torch.zeros(1,args.num_classes).cuda()
        with torch.no_grad():
            target = Variable(target.long().cuda())
            for i in range(deepen_times):
                # print('ensemble model:{}'.format(i))
                pred_i, _ = models[i](data, target)
                pred += pred_i
            pred = pred/deepen_times
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)

            list_acc.append((predicted == target).sum().item())
            prediction_list.append(predicted.cpu().detach().float().numpy())
            label_list.append(target.cpu().detach().float().numpy())
            correct_list.append(correct.cpu().detach().float().numpy())

            # label_list = F.one_hot(target, num_classes=args.num_classes).cpu().detach().float().numpy()
            probability = torch.softmax(pred, dim=1).cpu().detach().float().numpy()
            probability_list.append(torch.softmax(pred, dim=1).cpu().detach().float().numpy()[:,1])
            one_hot_probability_list.append(torch.softmax(pred, dim=1).squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(pred.data.squeeze(dim=0).cpu().detach().float().numpy())
            one_hot_label = F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy()
            one_hot_label_list.append(one_hot_label)
            ece_list.append(cal_ece(torch.squeeze(pred), target))
            # NLL brier
            nll, brier = calc_nll_brier(probability, pred, target, one_hot_label)
            nll_list.append(nll)
            brier_list.append(brier)
    logging.info('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    print('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    if args.num_classes > 2:
        epoch_auc = metrics.roc_auc_score(one_hot_label_list, one_hot_probability_list, multi_class='ovo')
    else:
        epoch_auc = metrics.roc_auc_score(label_list, probability_list)
        # epoch_auc = metrics.roc_auc_score(label_list, probability_list)
    # fpr, tpr, thresholds = roc_curve(label_list, probability_list, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    avg_acc = correct_num/data_num
    avg_ece = sum(ece_list)/len(ece_list)
    # epoch_auc = metrics.roc_auc_score(label_list, prediction_list)
    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    F1_Score = f1_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    Recall_Score = recall_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    aurc, eaurc = calc_aurc_eaurc(probability_list, correct_list)



    if not os.path.exists(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder))):
        os.makedirs(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder)))

    avg_nll = sum(nll_list) / len(nll_list)
    avg_brier = sum(brier_list) / len(brier_list)
    with open(os.path.join(args.save_dir, "{}_{}_{}_Metric.txt".format(args.model_name, args.dataset, args.folder)),
              'w') as Txt:
        Txt.write(
            "Acc: {}, AUC: {}, AURC: {}, EAURC: {},  NLL: {}, BRIER: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
                round(avg_acc, 6), round(epoch_auc, 6), round(aurc, 6), round(eaurc, 6), round(avg_nll, 6),
                round(avg_brier, 6), round(F1_Score, 6), round(Recall_Score, 6), round(avg_kappa, 6),
                round(avg_ece, 6)
            ))

    return avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modal_number', type=int, default=2, metavar='N',
                        help='modalties number')
    parser.add_argument("--checkpoint",type = str)

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--test_epoch', type=int, default=198, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda_epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument("--model_name", default="IMDR", type=str, help="Base_transformer/ResNet3D/Res2Net2D/Multi_ResNet/Multi_dropout_ResNet/Multi_DE_ResNet/Multi_CBAM_ResNet/Multi_EF_ResNet")
    parser.add_argument("--dataset", default="MMOCTF", type=str, help="MMOCTF/MGamma/Gamma/OLIVES")
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument("--mode", default="test", type=str, help="train/test/train&test")
    parser.add_argument("--model_base", default="transformer", type=str, help="transformer/cnn")

    #control Noise
    parser.add_argument("--condition", default="noise", type=str, help="noise/normal")
    parser.add_argument("--condition_name", default="Gaussian", type=str, help="Gaussian/SaltPepper/All")
    parser.add_argument("--Condition_SP_Variance", default=0.005, type=int, help="Variance: 0.01/0.1")
    parser.add_argument("--Condition_G_Variance", default=0.05, type=float, help="Variance: 15/1/0.1")

    # control log 
    parser.add_argument('--name', default='checkpoint_0.3',
                        type=str)  # name  Save


    args = parser.parse_args()
    args.seed_idx = 11


    Condition_G_Variance = [0.1,0.2,0.3,0.4,0.5]

    if args.dataset =="dr2":
        from DR_2.data_harvard import GAMMA_dataset as GAMMA_dataset_dr2
        
        args.modalties_name = ["FUN", "OCT"]
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.num_classes = 2
        args.modalties = len(args.dims)

        #Data Path
        args.base_path = 'Your_data_path'
        args.data_path = 'Your_train_path'

        filelists = os.listdir(args.data_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=10)

        y = kf.split(filelists)
        count = 0
        train_filelists = [[], [], [], [], []]
        val_filelists = [[], [], [], [], []]
        for tidx, vidx in y:
            train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
            count = count + 1
        f_folder = int(args.folder[-1])
        print(f"the folder is {f_folder}")

        train_dataset = GAMMA_dataset_dr2(args, dataset_root=args.data_path,
                                      oct_img_size=args.dims[0],
                                      fundus_img_size=args.dims[1],
                                      mode='train',
                                      label_file=args.base_path + 'train_839.xlsx',
                                      filelists=np.array(train_filelists[f_folder]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,num_workers=8)

        val_dataset = GAMMA_dataset_dr2(args, dataset_root=args.data_path,
                                    oct_img_size=args.dims[0],
                                    fundus_img_size=args.dims[1],
                                    mode='val',
                                    label_file=args.base_path + 'train_839.xlsx',
                                    filelists=np.array(val_filelists[f_folder]), )


        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, drop_last=True,num_workers=8)
        test_loader = val_loader
    elif args.dataset =="glu2":
        from glu2.data_glu2 import GAMMA_dataset as GAMMA_dataset_glu2
        import pandas as pd
        args.modalties_name = ["FUN", "OCT"]
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.num_classes = 2
        args.modalties = len(args.dims)

        #Data Path
        args.base_path = 'Your_data_path'
        args.data_path = 'Your_train_path'

        filelists = os.listdir(args.data_path)

        kf = KFold(n_splits=5, shuffle=True, random_state=10)

        y = kf.split(filelists)
        count = 0
        train_filelists = [[], [], [], [], []]
        val_filelists = [[], [], [], [], []]
        for tidx, vidx in y:
            train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
            count = count + 1
        f_folder = int(args.folder[-1])
        label_file=args.base_path + 'train.xlsx'
        df = pd.read_excel(label_file)
        #print(train_filelists[f_folder])
        data_list = df['data'].astype(str).values
        data_list = np.array([str(x).zfill(5) for x in data_list])

        # 按80%分割数据为训练集和验证集
        train_size = int(len(data_list) * 0.8)
        train_list = data_list[:train_size]
        val_list = data_list[train_size:]
        train_dataset = GAMMA_dataset_glu2(args, dataset_root=args.data_path,
                                      oct_img_size=args.dims[0],
                                      fundus_img_size=args.dims[1],
                                      mode='train',
                                      label_file=label_file,
                                      filelists=np.array(train_list))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,num_workers=8)

        val_dataset = GAMMA_dataset_glu2(args, dataset_root=args.data_path,
                                    oct_img_size=args.dims[0],
                                    fundus_img_size=args.dims[1],
                                    mode='val',
                                    label_file=label_file,
                                    filelists=np.array(val_list), )


        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, drop_last=True,num_workers=1)
        test_loader = val_loader




    else:
        print('There is no this dataset name')
        raise NameError

    # Baseline and  IMDR
    if args.model_name =="ResNet3D":
        args.modalties_name = ["OCT"]
        args.modal_number = 1
        args.dims = [(128, 256,128)]
        model = ResNet3D(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Res2Net2D":
        args.modalties_name = ["FUN"]
        args.modal_number = 1
        args.dims = [(512, 512)]
        args.modalties = len(args.dims)
        model = Res2Net2D(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)

    elif args.model_name =="Fusion_transformer":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = EyeMost_Plus_transformer(args.num_classes, args.modal_number, args.dims, args)

    elif args.model_name =="Multi_EF_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        if args.dataset == 'OLIVES':
            args.dims = [[(48+3, 248, 248)], [(512, 512)]] # OLIVES
        else:
            args.dims = [[(128 + 3, 256, 128)], [(512, 512)]]  # Our
        args.modalties = len(args.dims)
        model = Multi_EF_(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_CBAM_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_CBAM_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_dropout_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_dropout_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE1_ResNet":
        args.lr = 0.0001
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE2_ResNet":
        args.lr = 0.0003
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE3_ResNet":
        args.lr = 0.001
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE4_ResNet":
        args.lr = 0.0002
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE5_ResNet":
        args.lr = 0.00001
        args.modalties_name = ["FUN", "OCT"]
        # args.modalties = len(args.dims)
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        # args.modalties = len(args.dims)
        models = []
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
        models.append(model)
        models.append(model)
        models.append(model)
        models.append(model)
        models.append(model)

    # Our  Model
    elif args.model_name == "IMDR":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        if args.dataset == 'dr2':
            from DR_2.fusion_net import IMDR
            model = IMDR(args.num_classes, args.modal_number, args.dims, args)
        elif args.dataset == 'glu2':
            from glu2.fusion_net import IMDR
            model = IMDR(args.num_classes, args.modal_number, args.dims, args)

    else:
        print('There is no this model name')
        raise NameError

    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    seed_num = list(range(1,11))


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    model.cuda()
    best_acc = 0
    loss_list = []
    acc_list = []
    if args.mode =='train&test':
        epoch = 0
        checkpoint = args.checkpoint
        test_loss, best_test_acc = test(epoch, test_loader, model, checkpoint)
        import pdb;
        pdb.set_trace()

        print('===========Train begining!===========')
        for epoch in range(args.start_epoch, args.end_epochs + 1):
            print('Epoch {}/{}'.format(epoch, args.end_epochs))
            epoch_loss = train(epoch, train_loader, model, best_acc=0.0)
            print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
            print('===========Val begining!===========')
            val_loss, best_acc = val(epoch,val_loader,model,best_acc)
            loss_list.append(epoch_loss.avg)
            acc_list.append(best_acc)
            # print('===========Test begining!===========')
            # test_loss, best_test_acc = test(epoch, test_loader, model)
        print('===========Test begining!===========')
        #test_loss, best_test_acc = test(epoch, test_loader, model)
        loss_plot(args, loss_list)
        metrics_plot(args, 'acc', acc_list)







