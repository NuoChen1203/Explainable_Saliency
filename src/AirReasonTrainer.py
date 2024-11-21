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
from src.utils.metrics import (
    cal_cc_score, cal_sim_score, cal_kld_score,
    cal_auc_score, cal_nss_score, add_center_bias
)

# Experiment configuration
experiment_name = "supp_test"
project_name = "final_supp"
root_dir = "./workdir/"

# Initialize Comet ML experiment
experiment = Experiment(
    api_key="",
    project_name=project_name,
)
experiment.set_name(experiment_name)

# Command line arguments
parser = argparse.ArgumentParser(description='ReasonNet Model Training for GQA')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--checkpoint_dir', type=str, default=None)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--topk', type=int, default=3)
args = parser.parse_args()

args.checkpoint_dir = os.path.join(root_dir, project_name, experiment_name)

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with different lengths.
    """
    unpacked_batch = list(zip(*batch))
    collated_items = [default_collate(items) for items in unpacked_batch[:-2]]
    second_last_item = list(unpacked_batch[-2])
    last_item = list(unpacked_batch[-1])
    return *collated_items, second_last_item, last_item

def adjust_learning_rate(optimizer, epoch):
    """
    Adjusts the learning rate based on the epoch number.
    """
    if epoch >= 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (0.1 ** (epoch / 2))

def compute_loss(prediction, attention_map, fixation_map):
    """
    Computes the combined loss using NSS, KLD, and CC metrics.
    """
    saliency_map = attention_map.to(prediction.dtype)
    fixation_map = fixation_map.to(prediction.dtype)
    loss = NSS(prediction, fixation_map) + KLD(prediction, saliency_map) + CC(prediction, saliency_map)
    return loss

def train_epoch(model, train_loader, optimizer, epoch, iteration, ss_rate, experiment):
    """
    Trains the model for one epoch.
    """
    model.train()

    for batch_idx, (images, questions, answers, ops, att, agg_att, fixations, scene_graphs, bboxes) in enumerate(train_loader):
        # Prepare data
        images = Variable(images).cuda()
        ops = Variable(ops).cuda()
        att = Variable(att).cuda()
        agg_att = Variable(agg_att).cuda()
        fixations = Variable(fixations).cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output_corr, output_incorr, valid_mask = model(
            images, questions, agg_att, scene_graphs, answers, bboxes,
            epoch=ss_rate, fixmap=fixations, topk=args.topk
        )

        # Compute loss
        loss_dict = {"agg_att": compute_loss(output_corr, agg_att[:, 0, :, :], fixations[:, 0, :, :])}
        total_loss = loss_dict["agg_att"]

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % 1 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}',
                {k: v.item() for k, v in loss_dict.items()}
            )
            experiment.log_metrics({k: v.item() for k, v in loss_dict.items()}, step=iteration)
        iteration += 1

    return iteration

def validate(model, val_loader, iteration, experiment):
    """
    Validates the model on the validation set.
    """
    model.eval()
    metrics = {
        'cc_score': [], 'sim_score': [], 'kld_score': [],
        'nss_score': [], 'auc_score': [],
        'auc_faith_score': [], 'aopc_faith_score': [], 'lodds_faith_score': []
    }

    with torch.no_grad():
        for batch_idx, (images, questions, answers, agg_att, fixations, scene_graphs, bboxes) in enumerate(val_loader):
            # Prepare data
            if isinstance(questions, list):
                questions = torch.stack(questions)
            images = Variable(images).cuda()
            questions = Variable(questions).cuda()
            agg_att = Variable(agg_att).cuda()
            fixations = Variable(fixations).cuda()

            # Forward pass
            output_corr, output_incorr, valid_mask = model(
                images, questions, agg_att, ss_rate=scene_graphs,
                ans=answers, bbox_list=bboxes, need_faith_metric=True,
                fixmap=fixations, topk=args.topk
            )

            # Extract saliency and fixation maps
            agg_att_np = agg_att.cpu().detach().numpy()
            fixations_np = fixations.cpu().detach().numpy()
            agg_att_corr = agg_att_np[:, 0, :, :]
            fix_corr = fixations_np[:, 0, :, :]

            # Faithfulness metrics
            if output_incorr is not None:
                faith_loss, faith_metrics = output_incorr[0], output_incorr[1]
                auc_list, aopc_list, lodds_list = faith_metrics

            # Compute evaluation metrics for each sample
            for j in range(len(agg_att_np)):
                pred_map = output_corr.squeeze(1)[j].cpu().detach().numpy()
                metrics['cc_score'].append(cal_cc_score(pred_map, agg_att_corr[j]))
                metrics['sim_score'].append(cal_sim_score(pred_map, agg_att_corr[j]))
                metrics['kld_score'].append(cal_kld_score(pred_map, agg_att_corr[j]))
                metrics['nss_score'].append(cal_nss_score(pred_map, fix_corr[j]))
                metrics['auc_score'].append(cal_auc_score(pred_map, fix_corr[j]))

                if output_incorr is not None:
                    metrics['auc_faith_score'].extend(auc_list)
                    metrics['aopc_faith_score'].extend(aopc_list)
                    metrics['lodds_faith_score'].extend(lodds_list)

    # Calculate mean metrics
    mean_metrics = {k: np.mean(v) for k, v in metrics.items() if v}

    # Print and log metrics
    for metric_name, metric_value in mean_metrics.items():
        print(f'{metric_name}: {metric_value}')
    experiment.log_metrics(mean_metrics, step=iteration)

    return mean_metrics.get('cc_score', 0)

def main():
    """
    Main function to train and validate the model.
    """
    # Initialize data loaders
    train_dataset = Batch_generator('train')
    val_dataset = Batch_generator('val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8,
        collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8,
        collate_fn=custom_collate_fn
    )

    # Initialize model and optimizer
    model = ReasonNet().cuda()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=1e-7
    )

    # Training loop
    print('Start training model')
    iteration, best_val_acc = 0, 0

    if args.mode != 'train':
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'model_best.pth')))
        validate(model, val_loader, iteration, experiment)
        return

    for epoch in range(args.epoch):
        # Adjust learning rate
        if epoch == 0:
            adjust_learning_rate(optimizer, 1)
        elif epoch > 2:
            adjust_learning_rate(optimizer, epoch - 1)

        # Train for one epoch
        iteration = train_epoch(model, train_loader, optimizer, epoch, iteration, epoch, experiment)
        # Validate the model
        current_val_acc = validate(model, val_loader, iteration, experiment)

        # Save checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if current_val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model_best.pth'))
            best_val_acc = current_val_acc
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model.pth'))

    experiment.end()

if args.mode == 'train':
    main()
