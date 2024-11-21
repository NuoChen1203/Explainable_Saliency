import sys
import os
import argparse
import numpy as np
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Local imports
from src.utils.loss import NSS, CC, KLD, cross_entropy
from src.dataset.dataloaderMy_air import Batch_generator
from src.model.ReasonNet_air import ReasonNet
from src.utils.metrics import (cal_cc_score, cal_sim_score, cal_kld_score, 
                             cal_auc_score, cal_nss_score, add_center_bias)

# Experiment configuration
name = "supp_test"
project_name = "final_use_supp"
root = "./workdir/"

# Initialize Comet ML
experiment = Experiment(
    api_key="5MDt5Crj6Oz6n6NSJaEtJgiGW",
    project_name=project_name,
)
experiment.set_name(name)

# Command line arguments
parser = argparse.ArgumentParser(description='ReasonNet (UpDown) Model for GQA')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--checkpoint_dir', type=str, default=None)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--topk', type=int, default=3)
args = parser.parse_args()

args.checkpoint_dir = os.path.join(root, project_name, name)

def custom_collate_fn(batch):
    unpacked_batch = list(zip(*batch))
    collated_items = [default_collate(items) for items in unpacked_batch[:-2]]
    last_item = list(unpacked_batch[-1])
    second_last_item = list(unpacked_batch[-2])
    return *collated_items, second_last_item, last_item

def adjust_learning_rate(optimizer, epoch):
    if epoch >= 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (0.1 ** (epoch/2))

def add_loss(pred, agg_att, fix):
    sal_map = agg_att.to(pred.dtype)
    fix = fix.to(pred.dtype)
    return NSS(pred, fix) + KLD(pred, sal_map) + CC(pred, sal_map)

def train(model, trainloader, optimizer, epoch, iteration, ss_rate, experiment):
    model.train()
    
    for batch_idx, (img, que, ans, op, att, agg_att, fix, cur_scene_graph, bbox) in enumerate(trainloader):
        # Prepare data
        img, que, ans = Variable(img).cuda(), que, ans
        op, att = Variable(op).cuda(), Variable(att).cuda()
        agg_att, fix = Variable(agg_att).cuda(), Variable(fix).cuda()
        
        # Forward pass
        optimizer.zero_grad()
        output_corr, output_incorr, valid_mask = model(img, que, agg_att, cur_scene_graph, ans, bbox, 
                                                      epoch=ss_rate, fixmap=fix, topk=args.topk)
        
        # Calculate loss
        loss_dict = {"agg_att": add_loss(output_corr, agg_att[:,0,:,:], fix[:,0,:,:])}
        loss = loss_dict["agg_att"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 1 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(img)}/{len(trainloader.dataset)} '
                  f'({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}', 
                  {k: v.item() for k, v in loss_dict.items()})
            experiment.log_metrics(loss_dict, step=iteration)
        iteration += 1
        
    return iteration

def test(model, valloader, iteration, experiment):
    model.eval()
    metrics = {'cc_score': [], 'sim_score': [], 'kld_score': [], 'nss_score': [], 
               'auc_score': [], 'auc_faith_score': [], 'aopc_faith_score': [], 
               'lodds_faith_score': []}
    idx_list = {}
    
    with torch.no_grad():
        for batch_idx, (img, que, ans, agg_att, fix, cur_scene_graph, bbox) in enumerate(valloader):
            # Prepare data
            if type(que) is list:
                que = torch.stack(que)
            img, que = Variable(img).cuda(), Variable(que).cuda()
            agg_att, fix = Variable(agg_att).cuda(), Variable(fix).cuda()
            
            # Forward pass
            output_corr, output_incorr, valid_mask = model(img, que, agg_att, ss_rate=cur_scene_graph, 
                                                         ans=ans, bbox_list=bbox, need_faith_metric=True, 
                                                         fixmap=fix, topk=args.topk)
            
            # Calculate metrics
            agg_att = agg_att.to(output_corr[-1].dtype).cpu().detach().numpy()
            fix = fix.cpu().detach().numpy()
            agg_att_corr = agg_att[:,0,:,:]
            fix_corr = fix[:,0,:,:]
            
            if output_incorr is not None:
                faithloss, faith_metric = output_incorr[0], output_incorr[1]
                auc_list, aopc_list, lodds_list = faith_metric
            
            for j in range(len(agg_att)):
                cur_pred = output_corr.squeeze(1)[j].cpu().detach().numpy()
                metrics['cc_score'].append(cal_cc_score(cur_pred, agg_att_corr[j]))
                metrics['sim_score'].append(cal_sim_score(cur_pred, agg_att_corr[j]))
                metrics['kld_score'].append(cal_kld_score(cur_pred, agg_att_corr[j]))
                metrics['nss_score'].append(cal_nss_score(cur_pred, fix_corr[j]))
                metrics['auc_score'].append(cal_auc_score(cur_pred, fix_corr[j]))
                
                if output_incorr is not None:
                    metrics['auc_faith_score'].extend(auc_list)
                    metrics['aopc_faith_score'].extend(aopc_list)
                    metrics['lodds_faith_score'].extend(lodds_list)
    
    # Log metrics
    mean_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    for k, v in mean_metrics.items():
        print(f'{k}: {v}')
    experiment.log_metrics(mean_metrics, step=iteration)
    
    return mean_metrics['cc_score']

def main():
    # Initialize data loaders
    train_data = Batch_generator('train')
    val_data = Batch_generator('val')
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                            shuffle=True, num_workers=8, 
                                            collate_fn=custom_collate_fn)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=8, 
                                          collate_fn=custom_collate_fn)
    
    # Initialize model and optimizer
    model = ReasonNet().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    
    # Training loop
    print('Start training model')
    iteration, val_acc = 0, 0
    
    for epoch in range(min(args.epoch, 7)):
        # Adjust learning rate
        if epoch == 0:
            adjust_learning_rate(optimizer, 1)
        elif epoch > 2:
            adjust_learning_rate(optimizer, epoch-1)
        
        # Train and validate
        iteration = train(model, trainloader, optimizer, epoch, iteration, epoch, experiment)
        cur_acc = test(model, valloader, iteration, experiment)
        
        # Save checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if cur_acc > val_acc:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model_best.pth'))
            val_acc = cur_acc
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model.pth'))
    
    experiment.end()

if args.mode == 'train':
    main()