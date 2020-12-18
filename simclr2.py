import cv2
import numpy as np
import torchvision.transforms as transforms
import os

from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

# comprises of train data loader, validation data loader and test data loader and the specific transforms for each
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule

# normalizations for each data module
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, stl10_normalization
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

from pl_bolts.models.self_supervised.resnets import resnet50
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.metrics import mean, accuracy

from pl_bolts.models.self_supervised.evaluator import Flatten
from pl_bolts.optimizers import LARSWrapper

train_data_dir = '/Users/johnathontoh/Desktop/contrastive_learning/coco_dataset_2014/train/'
val_data_dir = '/Users/johnathontoh/Desktop/contrastive_learning/coco_dataset_2014/val/'
test_data_dir = '/Users/johnathontoh/Desktop/contrastive_learning/coco_dataset_2014/test/'


#--------- Gaussian Blur Technique ---------------
class GaussianBlur(object):
    # Implements Gaussian Blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample




#--------- Implement Data Transformations ----------------
class SimCLRTrainDataTransforms(object):

    # this function returns none
    # the -> None just tells that f() returns None (but it doesn't force the function to return None)
    def __init__(self,
                 # image size
                 input_height: int = 224,

                 # using gaussian blur for ImageNet, may not need for other dataset
                 gaussian_blur: bool = False,

                 # using jitter for ImageNet, may not need for other dataset
                 jitter_strength: float = 1.,

                 # normalization
                 normalize: Optional[transforms.Normalize] = None) -> None:

        # track the variables internally
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize

        # jitter transform
        # from SimCLR paper
        # apply to only PIL images
        self.colour_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        # can apply to the same image and will get a different result every single time
        # apply to only PIL images
        data_transforms = [
            transforms.RandomResizedCrop(self.input_height),
            # p refers to the probability
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.colour_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        # apply to only PIL images
        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.input_height, p=0.5)))

        # apply all the transformations to tensors when working in pytorch
        data_transforms.append(transforms.ToTensor())

        # normalization
        if self.normalize:
            data_transforms.append(normalize)

        # transforms.Compose just clubs all the transforms provided to it.
        # So, all the transforms in the transforms.Compose are applied to the input one by one.
        self.train_transform = transforms.Compose(data_transforms)

    #  The __call__ method enables Python programmers to write classes where the instances behave like functions and can be called like a function
    # sample refers to an input image
    def __call__(self, sample):

        # call the instance self.train_transform and make it behave like a function
        # use the train_transform as specified in the initialization
        transform = self.train_transform

        # apply the transformation twice to 2 version of the same image as specified in the simclr paper
        xi = transform(sample)  # first version
        xj = transform(sample)  # second version

        return xi,xj


# -------- evaluation of the the data transformation ---------
class SimCLREvalDataTransform(object):
    # track these parameters internally
    def __init__(self,
                 input_height: int = 224,
                 normalize: Optional[transforms.Normalize] = None
                 ):

        self.input_height = input_height
        self.normalize = normalize

        # convert to tensor
        eval_data_transforms = [
            transforms.Resize(self.input_height),
            transforms.ToTensor()
        ]

        if self.normalize:
            eval_data_transforms.append(normalize)

        self.test_transforms = transforms.Compose(eval_data_transforms)

    # take the same input image used in the training, "sample"
    def __call__(self, sample):

        # call the instance self.test_transforms and make it behave like a function
        transform = self.test_transforms

        xi = transform(sample)  # first version
        xj = transform(sample)  # second version

        return xi, xj


#----------- Implement projection head ----------------------

# projection layer at the end of the model
# map from the output of the encoder into the dimension that we want to have for the comparison
class Projection(nn.Module):
    def __init__(self, input_dim = 2048, hidden_dim = 2048, output_dim = 128):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x,dim=1)


# normalized temperature-scaled cross entropy loss
# output 1 and output 2 is the 2 different versions of the same input image
def nt_xent_loss(output1, output2, temperature):
    # concatenate v1 img and v2 img via the rows, stacking vertically
    out = torch.cat([output1, output2], dim=0)
    n_samples = len(out)


    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    # the diagonal of the matrix is the square of each vector element in the out vector, which shows the similarity between the same elements
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov/temperature)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    neg = sim.masked_select(mask).view(n_samples,-1).sum(dim=-1)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    pos = torch.exp(torch.sum(output1 * output2, dim=-1)/temperature)
    # concatenate via the rows, stacking vertically
    pos = torch.cat([pos,pos], dim=0)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos/neg).mean()
    return loss

# attach a small MLP to the outputs of the encoder
# drop_p --> dropout probability of the MLP
# hidden_dim --> hidden layers of the MLP
# z_dim --> projection dimension of the encoder
# num_classes --> needed to build the MLP
class SSLOnlineEvaluator(pl.Callback):
    def __init__(self, drop_p: float=0.2, hidden_dim: int=1024, z_dim: int=None, num_classes: int=None):
        super().__init__()
        self.drop_p = drop_p
        self.hidden_dim = hidden_dim
        self.optimizer = None
        self.z_dim = z_dim
        self.num_classes = num_classes

    # before training happens
    # before the summary prints out
    # built-in simple MLP into bolts, 1 layer and then dropouts then output
    # to(pl_module.device) --> not part of the lightning module so have to add to the gpu
    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
        # takes an input --> flatten it --> add dropouts --> put a linear layer
        # simple linear mapping
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes = self.num_classes,
            p=self.drop_p,).to(pl_module.device)
        # pass in the parameters of the non_linear_evaluator and use SGD
        # need to learn the weights for the mlp_loss
        # need an optimizer
        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=1e-3)

    # once the model has been updated
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        (x1, _), y = batch
        x1 = x1.to(pl_module.device)
        y = y.to(pl_module.device)

        # extract representations from the model with no backprobagation to the model
        with torch.no_grad():
            representations = pl_module(x1)

        # forward
        # mlp predictions
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds,y)

        # update fine tune weights
        # take the loss and do backward probagation
        mlp_loss.backward()
        # apply the optimizer step
        self.optimizer.step()
        # zero out the gradient
        self.optimizer.zero_grad()

        # the mlp prediction and the labels
        # log the accuracy
        acc = accuracy(mlp_preds, y)
        metrics = {'ft_callbacks_mlp_loss': mlp_loss, 'ft_callbacks_mlp_acc': acc}

        # log the metrics
        # global_step --> only training steps, not validation steps
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)




# using pytorch lightning
# it would be nn.Module if it was just the normal pytorch
class SimCLR(pl.LightningModule):
    def __init__(self,

                 batch_size,

                 num_samples,

                 # number of GPU
                 world_size=1,

                 warmup_epochs = 10,

                 max_epochs = 100,

                 # LARS (Layer-wise Adaptive Rate Scaling) Learning Rate
                 lars_lr = 0.1,

                 lars_eta = 1e-3,

                 # optimised weight decay
                 opt_weight_decay = 1e-6,

                 loss_temperature = 0.5
                 ):
        super().__init__()

        # for pytorch lightning, it saves all the __init__ parameters to the checkpoint
        # save all the arguments into save_hyperparameters()
        self.save_hyperparameters()

        # predefined above
        self.nt_xent_loss = nt_xent_loss
        self.encoder = resnet50()
        # Projection Layer
        # h --> || --> z (non-linear)
        self.projection = Projection()

        # replace the first conv layer to make it work for cifar10
        # only needed because CIFAR-10 images will shrink too much
        # the original resnet50 has a kernel_size of 7 and stride of 2 in the first conv layer
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # original resnet50 --> kernel_size=2, stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

    # output of the encoder
    def forward(self, x):
        result = self.encoder(x)[-1]
        return result

    # ignore the bias and the batch norm
    # do not apply weight decay to the bias and batch norm
    # do not want to constraint the bias (e.g. controls the shift of a line on the y-axis for 1D) using the L2 regularization
    # the batch norm parameters controls the scale and shift of the output of the batch norm layer --> do not want to restrict those scale and shift values
    def exclude_from_wt_decay(self,named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params':params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]

    # for each gpu in the model, it will call setup in that model
    def setup(self, stage):
        # get each batch_size for each gpu
        # global_batch_size --> total batch size for all the gpus the model is training on
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        # the floor division "//" rounds the result down to the nearest whole number
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def configure_optimizers(self):
        # exclude certain parameters
        # ignore from the weight_decay, all the parameters
        # it looks through all the parameters in the model (e.g. encoder, projection) and do not apply weight decay to the bias and batch norm
        parameters = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.hparams.opt_weight_decay)

        # the parameters do not include the bias or batch norm
        # TRICK 1 --> use LARS + filter weights
        optimizer = torch.optim.SGD(parameters, lr=self.hparams.lars_lr)
        optimizer_LARS = LARSWrapper(optimizer, eta=self.hparams.lars_eta)

        # TRICK 2 --> after each step
        # After optimizer is defined, the scheduler is then defined
        # The scheduler is used after each step (also known as iterations) --> warm_up_epochs x train_iter = total number of steps for warm_ups
        # update the learning rate every training steps (training iterations)
        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        max_epochs = self.trainer.max_epochs * self.train_iters_per_epoch

        # the scheduler, which is perform each step
        # from pl_bolts
        # the scheduler takes all these parameters
        # from the warmup_start_lr --> max learning rate of optimizer and the number of epochs spcified for warmups --> cosine decay for the remainder of the epochs
        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0,
            # final learning rate
            eta_min=0
        )

        # use a dictionary to define the scheduler for each step (pytorch lightning)
        # default pytorch lightning updates scheduler every epoch, can overwrite it as shown below
        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            # every 1 step
            # if value change to 5, it means the scheduler will update every 5 steps
            'frequency': 1
        }

        # return an array because you can have multiple optimizers or schedulers
        return [optimizer], [scheduler]

    # pass in optimizer argument if there are multiple optimizers e.g def training_step(self, batch, batch_idx, optimizer_idx)
    def shared_step(self, batch, batch_idx):
        # 2 versions of the image
        # y is the label
        (img1, img2), y = batch

        # ENCODE
        # the anchor for img1
        # in bolts, the torchvision resnet is modified to return the output of every single layer (each resnet block)
        # this takes the last output
        # encode --> representations
        h1 = self.encoder(img1)[-1]
        h2 = self.encoder(img2)[-1]

        # PROJECTION
        # img --> encoder --> result is h --> || (projection head) --> z
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)

        # log the result
        result = pl.TrainResult(minimize=loss)
        # log the loss for each training step and get the mean loss for each epoch
        result.log('train_loss', loss, on_epoch=True)
        return result

    # to check how the training is doing
    # anything under eval has no gradient
    """
    e.g.
    model.eval()
    with torch.no_grad():
        loss = model.validation_step(batch)
    # after the validation is done, the model will train again
    model.train()
    
    Therefore, a shared function will not matter
    """
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # checkpoint on the loss value
        result = pl.EvalResult(checkpoint_on=loss)
        # log the loss for each epoch only, you only want to validate for each epoch
        result.log('avg_val_loss', loss, on_epoch=True, on_step=False)
        return result


if __name__ == '__main__':
    # random crop and resize of 32 for training image
    coco_height = 224
    batch_size = 1024

    # init data
    #dm = CIFAR10DataModule(os.getcwd(), num_workers=8, batch_size=batch_size)
    coco_train = DataLoader(train_data_dir, batch_size=batch_size, num_workers=8)
    coco_train.train_transforms = SimCLRTrainDataTransforms(coco_height)

    coco_val = DataLoader(val_data_dir, batch_size=batch_size, num_workers=8)
    coco_val.val_transforms = SimCLREvalDataTransform(coco_height)
    # dm.test_transforms = SimCLREvalDataTransform(cifar_height)



    # callbacks
    lr_monitor = LearningRateMonitor()
    finetuner = SSLOnlineEvaluator(z_dim=2048*2*2, num_classes=80)
    callbacks = [lr_monitor, finetuner]



    # get number of samples
    train_samples = len(os.listdir(train_data_dir))

    # init model
    model = SimCLR(batch_size = batch_size, num_samples = train_samples)

    # train the model
    # progress_bar_refresh_rate --> update every 2 batches
    # sync_batchnorm sync all the batch norm for all gpus
    trainer = pl.Trainer(callbacks=callbacks, progress_bar_refresh_rate=2, gpus=1, precision=16, sync_batchnorm=True)
    trainer.fit(model, coco_train)














