
import os
import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm
from lib.losses.si_log import SILogLoss
from lib.losses.bins_chemfer import BinsChamferLoss

LOSS_SCHEDULING_COMPOSITE = 'composite'
LOSS_SCHEDULING_RANDOM = 'random'

class ModelTrainer():
    """Helper class for model training with Tensorboard.

    Available loss handling techniques:
         ¤ LOSS_SCHEDULING_COMPOSITE: uses the weighted sum of all losses.
         ¤ LOSS_SCHEDULING_RANDOM: oscillates between segmentation loss and depth prediction loss.
    """
    def __init__(
        self,
        model,
        optimizer,
        min_depth,
        max_depth,
        device,
        loss_scheduling = LOSS_SCHEDULING_COMPOSITE,
        beta = 0.1
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.beta = beta
        self.loss_scheduling = loss_scheduling
        self.best_loss = float('inf')

        self.criterion_pixels = SILogLoss()
        self.criterion_bins = BinsChamferLoss()
        self.criterion_seg = torch.nn.CrossEntropyLoss()

    def get_depth_losses(self, pred_depth, true_depth, bin_centers, mask):
        l_pixel = self.criterion_pixels(pred_depth, true_depth, mask = mask, interpolate = True)
        l_bins = self.criterion_bins(bin_centers, true_depth.float())

        return l_pixel, l_bins

    def epoch(self, data_loader, is_train = True):
        """Runs one epoch of training/ validation on the data loader.

        Args:
            data_loader: data loader with batched data points.
            is_train (bool, optional): signifies if it's a training or validation epoch. Defaults to True.

        Returns:
            (pixel loss value, bins loss value, segmentation loss value, total loss value)
        """
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        
        loss_value_pixel = 0
        loss_value_bins = 0
        loss_value_seg = 0
        loss_value_total = 0

        for data in tqdm(data_loader):
            inputs, targets = data
            seg_target = targets[1]
            depth_target = targets[0]

            batch_size = len(inputs)
            inputs = inputs.to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            # predict
            bin_centers, pred_depth, pred_seg = self.model(inputs)

            pred_seg = F.interpolate(
                input = pred_seg,
                size = inputs.shape[-2:],
                mode = 'bilinear',
                align_corners = True
            )
            
            mask = depth_target > self.min_depth
            seg_target = seg_target.squeeze(1).to(self.device)
            depth_target = depth_target.to(self.device)

            # compute loss
            if self.loss_scheduling == LOSS_SCHEDULING_COMPOSITE:
                l_pixel = self.criterion_pixels(pred_depth, depth_target, mask = mask, interpolate = True)
                l_bins = self.criterion_bins(bin_centers, depth_target.float())

                l_seg = self.criterion_seg(pred_seg, seg_target.long())
                l_total = 0.5 * (l_pixel + self.beta * l_bins) + 0.5 * l_seg
                
                loss_value_pixel += l_pixel.data.item() / batch_size
                loss_value_bins += l_bins.data.item() / batch_size
                loss_value_seg += l_seg.data.item() / batch_size

            elif self.loss_scheduling == LOSS_SCHEDULING_RANDOM:
                if np.random.choice([True, False]):
                    l_pixel = self.criterion_pixels(pred_depth, depth_target, mask = mask, interpolate = True)
                    l_bins = self.criterion_bins(bin_centers, depth_target.float())
                    
                    l_total = l_pixel + self.beta * l_bins

                    loss_value_pixel += l_pixel.data.item() / batch_size
                    loss_value_bins += l_bins.data.item() / batch_size
                else:
                    l_total = self.criterion_seg(pred_seg, seg_target.long())

                    loss_value_seg += l_total.data.item() / batch_size

            else:
                assert 'Unknown loss handling!'

            loss_value_total += l_total.data.item() / batch_size

            # learn
            if is_train:
                l_total.backward()
                self.optimizer.step()

        dataset_size = len(data_loader)
        loss_value_pixel = loss_value_pixel / dataset_size
        loss_value_bins = loss_value_bins / dataset_size
        loss_value_seg = loss_value_seg / dataset_size
        loss_value_total = loss_value_total / dataset_size

        return loss_value_pixel, loss_value_bins, loss_value_seg, loss_value_total

    def train(self, train_loader, test_loader, total_epochs, log_writter, save_dir):
        """Trains and validates model for 'total_epochs'.

        Args:
            train_loader: data loader for training
            test_loader: data loader for testing
            total_epochs: total epochs in training
            log_writter: Tensorboard summary writer
            save_dir: path to save the best model
        """
        for i in range(total_epochs):
            # training epoch
            loss_value_pixel, loss_value_bins, loss_value_seg, loss_value_total = self.epoch(train_loader, is_train = True)

            log_writter.add_scalar('pixel_loss/train', loss_value_pixel, i)
            log_writter.add_scalar('bin_loss/train', loss_value_bins, i)
            log_writter.add_scalar('seg_loss/train', loss_value_seg, i)
            log_writter.add_scalar('total_loss/train', loss_value_total, i)
            
            print('epoch ', i, ' -> Train loss: ', loss_value_total)

            # validation epoch
            with torch.no_grad():
                loss_value_pixel, loss_value_bins, loss_value_seg, loss_value_total = self.epoch(test_loader, is_train = False)
                
            log_writter.add_scalar('pixel_loss/val', loss_value_pixel, i)
            log_writter.add_scalar('bin_loss/val', loss_value_bins, i)
            log_writter.add_scalar('seg_loss/val', loss_value_seg, i)
            log_writter.add_scalar('total_loss/val', loss_value_total, i)
            
            print('epoch ', i, ' -> Val loss: ', loss_value_total)

            if self.best_loss > loss_value_total:
                self.best_loss = loss_value_total
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best.ckpt'))
