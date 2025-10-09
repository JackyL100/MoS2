import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
import time
import os

class Tester:
    def __init__(self,
                 loss_fn = None,
                 optimizer = None,
                 lr: float = None,
                 epochs: int = None,
                 train_data = None,
                 test_data = None,
                 record_file = None,
                 color_mapping: torch.Tensor = None):
        '''
        Initializes Tester object

        Args:
        loss_fn (function): function that calculates loss given predictions and targets
        optimizer (Python optimizer): optimizer to be used for model training (not used yet)
        lr (float): learning rate for optimizer
        epochs (int): how many epochs to train model on train_data
        train_data (Pytorch Dataloader): dataloader containing training data consisting of the image batch, corresponding binary masks for each class, composite mask
        test_data (Pytorch Dataloader): dataloader containing testing data consisting of the image batch, corresponding binary masks for each class, composite mask
        '''
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data
        self.record = None
        self.color_mapping = torch.tensor([(0,0,0), (6,4,243), (88,255,52), (255,28,36), (77,241,232), (209,209,216)]) if color_mapping == None else color_mapping

        if record_file is not None:
            if os.path.exists(os.path.dirname(record_file)):
                  self.record = open(record_file)

    def train(self, model, optimizer):
        '''
        Train model according to Tester's configuration.
        
        Args:
            model (Pytorch model): model to be trained 
            optimizer (Pytorch optimizer): optimizer tied to model's parameters, will be replaced with class attribute later
            
        Returns:
            trained model
            training information'''
        model.train()
        losses = []
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0
            for batch in self.train_data:
                            images, masks, _ = batch
            segmented = model(images)
            loss = self.loss_fn(segmented, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().item()

    def _dice_score(y_true: torch.Tensor, y_pred: torch.Tensor):
        """ 
        Computes dice score between rgb masks 

        Args:
            y_true (Pytorch tensor): tensor of shape (batch, 3, height, width) or (batch, height, width, 3) containing ground truth
            y_pred (Pytorch tensor): tensor of shape (batch, 3, height, width) or (batch, height, width, 3) containing binary masks from model output

        Returns:
            dice_score (float array): array of length batch with dice_score[i] between 0 and 1 representing the dice score between y_true[i] and y_pred[i]
        """
        assert y_true.shape == y_pred.shape, f'Inputs shapes not equal. y_true.shape: {y_true.shape} | y_pred.shape: {y_pred.shape}'

        # if (batch, 3, height, width), then permute to (batch, height, width, 3) for easier for loop
        # function takes in (batch, 3, height, width) because it is a common format
        if y_true.shape[1] == 3:
            y_true = y_true.permute(0, 2, 3, 1)
        if y_pred.shape[1] == 3:
            y_pred = y_pred.permute(0, 2, 3, 1)
        dice_score = []
        for mask in range(y_true.shape[0]):
            overlap = (y_pred[mask] == y_true[mask]).all(dim=2).flatten().sum()
            total_area = y_true.shape[1] * y_true.shape[2] * 2
            dice_score.append((2 * overlap) / total_area)
        return dice_score
    
    
    def _assemble_image(self, segmented):
        '''
        Assemble segmented images from segmentation model's output

        Args:
            segmented (Pytorch tensor): (1,6,568,764) tensor that contains 6 binary masks, one for each class

        Returns:
            RGB image made of combined binary masks
        '''
        indices = torch.argmax(segmented, dim=1,keepdim=True)
        return color_mapping[indices][0][0]

    def evaluate(self, model):
        '''
        Test model on testing data

        Args:
            model (Pytorch model): model to be tested for evaluation and comparison
        Returns:
            binary masks (Pytorch Tensor): binary masks for each class in segmentation containing probability that each pixel belongs to the class | shape (batch size, number of classes, height, width)
            assembled image (Pytorch Tensor): composite image of color masks overlayed on original image, labeling each pixel with a class
            average dice score (float): average dice score of composite image across whole batch
        '''

        avg_batch_dice_score = 0
        i = 0
        for batch in self.test_data:
            i += 1
            img, masks, mask = batch
            output_binary_masks = model(img)
            softmax = torch.nn.Softmax2d()

            # normalize to get all masks within the same range
            # only use on first batch for now
            output_binary_masks = torch.nn.functional.normalize(output_binary_masks[0], 1, dim=(2,3))
            # softmax to get values in interval (0,1)
            output_binary_masks = softmax(output_binary_masks)

            assembled_image = self._assemble_image(output_binary_masks)
            avg_dice_score += self._dice_score(mask, assembled_image)
        return output_binary_masks, assembled_image, avg_batch_dice_score / i, 
