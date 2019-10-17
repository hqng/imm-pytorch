"""
Train Evaluate and Test Model
"""
import argparse
import time
import os
from os import path
import gc
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.utils.data import DataLoader
from visdom import Visdom
from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomSaver

import data
from imm_model import AssembleNet
from criterion import LossFunc
import utils

PARSER = argparse.ArgumentParser(description='Option for Glaucoma')
#------------------------------------------------------------------- data-option
PARSER.add_argument('--data_root', type=str,
                    default='../data/',
                    help='location of root dir')
PARSER.add_argument('--dataset', type=str,
                    default='celeba',
                    help='location of dataset')
PARSER.add_argument('--testset', type=str,
                    default='../data/',
                    help='location of test data')
PARSER.add_argument('--nthreads', type=int, default=8,
                    help='number of threads for data loader')
PARSER.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='train batch size')
PARSER.add_argument('--val_batch_size', type=int, default=8, metavar='N',
                    help='val batch size')
#------------------------------------------------------------------ model-option
PARSER.add_argument('--pretrained_model', type=str, default='',
                    help='pretrain model location')
PARSER.add_argument('--loss_type', type=str, default='perceptual',
                    help='loss type for criterion: perceptual | l2')
#--------------------------------------------------------------- training-option
PARSER.add_argument('--seed', type=int, default=1234,
                    help='random seed')
PARSER.add_argument('--gpus', type=list, default=[],
                    help='list of GPUs in use')
#optimizer-option
PARSER.add_argument('--optim_algor', type=str, default='Adam',
                    help='optimization algorithm')
PARSER.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
PARSER.add_argument('--weight_decay', type=float, default=1e-8,
                    help='weight_decay rate')
#saving-option
PARSER.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
PARSER.add_argument('--checkpoint_interval', type=int, default=1,
                    help='epoch interval of saving checkpoint')
PARSER.add_argument('--save_path', type=str, default='checkpoint',
                    help='directory for saving checkpoint')
PARSER.add_argument('--resume_checkpoint', type=str, default='',
                    help='location of saved checkpoint')
#only prediction-option
PARSER.add_argument('--trained_model', type=str, default='',
                    help='location of trained checkpoint')

ARGS = PARSER.parse_args()

DEVICE = torch.device('cuda:{}'.format(ARGS.gpus[0]) if len(ARGS.gpus) > 0 else 'cpu')
# Set the random seed manually for reproducibility.
torch.manual_seed(ARGS.seed)
if DEVICE.type == 'cuda':
    cuda.set_device(ARGS.gpus[0])
    cuda.manual_seed(ARGS.seed)

class Main():
    """Wrap training and evaluating processes
    """
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(self.opt.save_path, exist_ok=True)

        self.neuralnet = self._make_model(opt)
        self.optimizer = self._make_optimizer(opt, self.neuralnet)
        self.train_loader, self.val_loader = self._make_data(opt)

        #loss function
        self.criterion = LossFunc(opt.loss_type)

        #batch data transform
        self.batch_transform = data.BatchTransform()

        #meter
        self.loss_meter = meter.AverageValueMeter()
        self.score_meter = meter.AverageValueMeter()
        # self.confusion_meter = meter.ConfusionMeter(opt.n_classes, normalized=True)

    def _make_model(self, opt):
        "create model, criterion"
        #if use pretrained model (load pretrained weight)
        neuralnet = AssembleNet()

        if opt.pretrained_model:
            print("Loading pretrained model {} \n".format(opt.pretrained_model))
            pretrained_state = torch.load(opt.pretrained_model, \
                    map_location=lambda storage, loc: storage, \
                    pickle_module=pickle)['modelstate']
            neuralnet.load_state_dict(pretrained_state)

        model_parameters = filter(lambda p: p.requires_grad, neuralnet.parameters())
        n_params = sum([p.numel() for p in model_parameters])
        print('number of params', n_params)

        return neuralnet

    def _make_optimizer(self, opt, neuralnet, param_groups=None):
        parameters = filter(lambda p: p.requires_grad, neuralnet.parameters())

        if param_groups is not None:
            lr = param_groups[0]['lr']
            weight_decay = param_groups[0]['weight_decay']
        else:
            lr = opt.lr
            weight_decay = opt.weight_decay

        optimizer = getattr(optim, self.opt.optim_algor)(
            parameters, lr=lr, weight_decay=weight_decay)

        return optimizer

    def _make_data(self, opt):
        #get data
        trainset = data.get_dataset(opt.data_root, opt.dataset, subset='train')
        valset = data.get_dataset(opt.data_root, opt.dataset, subset='val')

        #dataloader
        train_loader = DataLoader(dataset=trainset, \
            num_workers=opt.nthreads, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=valset, \
            num_workers=opt.nthreads, batch_size=opt.val_batch_size, shuffle=False)

        return train_loader, val_loader

    #===========================================================================
    # Training and Evaluating
    #===========================================================================

    def _resetmeter(self):
        self.loss_meter.reset()
        self.score_meter.reset()
        # self.confusion_meter.reset()

    def _evaluate(self, dataloader):
        gc.collect()
        self.neuralnet.eval()
        self._resetmeter()

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                im = batch[0].requires_grad_(False).to(DEVICE)
                keypts = batch[1].requires_grad_(False).to(DEVICE)

                im, future_im, mask, _, _ = self.batch_transform.exe(im, landmarks=keypts)

                future_im_pred, gauss_yx, pose_embeddings = self.neuralnet(im, future_im)

                #loss
                loss = self.criterion(future_im_pred, future_im, mask)

                #log meter
                self.loss_meter.add(loss.item())
                # self.score_meter.add(jaccscore.item())

        self.neuralnet.train()
        return self.loss_meter.value()[0]

    def _train(self, dataloader, epoch):
        self.neuralnet.train()
        self._resetmeter()

        for iteration, batch in enumerate(dataloader, 1):
            start_time = time.time()

            im = batch[0].requires_grad_(False).to(DEVICE)
            keypts = batch[1].requires_grad_(False).to(DEVICE)

            im, future_im, mask, _, _ = self.batch_transform.exe(im, landmarks=keypts)

            future_im_pred, gauss_yx, pose_embeddings = self.neuralnet(im, future_im)

            #loss
            loss = self.criterion(future_im_pred, future_im, mask)

            loss.backward()
            self.optimizer.step()

            #log meter
            self.loss_meter.add(loss.item())
            # self.score_meter.add(jaccscore.item())

            #print
            eslapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} ith_batch | time(s) {:5.2f} | \
                conten loss {:5.2f} | reconstruction loss {:5.2f}'.format( \
                    epoch, iteration, len(dataloader), eslapsed, \
                    content_loss.item(), reconstruction_loss.item()))

        return self.loss_meter.value()[0]

    def exe(self):
        print(self.opt)
        print('\n')
        start_epoch = 1
        best_result = 1.
        best_flag = False

        #resume from saved checkpoint
        if self.opt.resume_checkpoint:
            print('Resuming checkpoint at {}'.format(self.opt.resume_checkpoint))
            checkpoint = torch.load(
                self.opt.resume_checkpoint,
                map_location=lambda storage, loc: storage, pickle_module=pickle)

            model_state = checkpoint['modelstate']
            self.neuralnet.load_state_dict(model_state)

            optim_state = checkpoint['optimstate']
            self.optimizer = self._make_optimizer(
                self.opt, self.neuralnet, param_groups=optim_state['param_groups'])

            start_epoch = checkpoint['epoch']+1
            best_result = checkpoint['best_result']

        #DataParallel for multiple GPUs:
        if len(self.opt.gpus) > 1:
            #dim always is 0 because of input data always is in shape N*W
            self.neuralnet = nn.DataParallel(self.neuralnet, device_ids=self.opt.gpus, dim=0)

        self.neuralnet.to(DEVICE)
        self.criterion.to(DEVICE)

        #visualization
        port = 8097
        viz = Visdom()
        visdom_saver = VisdomSaver([viz.env])

        loss_logger = VisdomPlotLogger('line', port=port, \
            opts={'title': 'Total Loss', 'legend': ['train', 'val']})

        losses_logger = VisdomPlotLogger('line', port=port, \
            opts={'title': 'Losses', 'legend': \
                ['train_reconst', 'train_percept', 'val_reconst', 'val_percept']})

        print('Start training optim {}, on device {}'.format( \
            self.opt.optim_algor, DEVICE.type))

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=5e-5)

        for epoch in range(start_epoch, self.opt.epochs+1):
            #update learning rate
            if self.opt.optim_algor != 'Adadelta':
                lr_scheduler.step()

            #let's go
            print('\n')
            print('-' * 65)
            print('{}'.format(time.asctime(time.localtime())))
            print(' **Training epoch {}, lr {}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            start_time = time.time()
            train_loss, train_reconst, train_percept = self._train(self.train_loader, epoch)

            print('| finish training on epoch {:3d} | time(s) {:5.2f} | loss {:3.4f}'.format(
                epoch, time.time() - start_time, train_loss))

            print(' **Evaluating on validate set')

            start_time = time.time()
            val_loss, val_reconst, val_percept = self._evaluate(self.val_loader)

            print('| finish validating on epoch {:3d} | time(s) {:5.2f} | loss {:3.4f}'.format(
                epoch, time.time() - start_time, val_loss))

            if val_loss < best_result:
                best_result = val_loss
                best_flag = True
                print('*' * 10, 'BEST result {} at epoch {}'.format(best_result, epoch), '*' * 10)

            if epoch % self.opt.checkpoint_interval == 0 or epoch == self.opt.epochs or best_flag:
                print(' **Saving checkpoint {}'.format(epoch))
                snapshot_prefix = path.join(self.opt.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_{}.pt'.format(epoch)

                model_state = self.neuralnet.module.state_dict() \
                    if len(self.opt.gpus) > 1 else self.neuralnet.state_dict()

                optim_state = self.optimizer.state_dict()
                checkpoint = {
                    'modelstate':model_state,
                    'optimstate':optim_state,
                    'epoch':epoch,
                    'best_result':best_result,
                    }
                torch.save(checkpoint, snapshot_path, pickle_module=pickle)

                #delete old checkpoint
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
                if best_flag:
                    best_prefix = path.join(self.opt.save_path, 'BEST')
                    best_path = best_prefix + '_{}.pt'.format(epoch)
                    torch.save(checkpoint, best_path, pickle_module=pickle)
                    best_flag = False
                    for f in glob.glob(best_prefix + '*'):
                        if f != best_path:
                            os.remove(f)
                print('| finish saving checkpoint {}'.format(epoch))

            #visualize training and eval process
            loss_logger.log((epoch, epoch), (train_loss, val_loss))
            losses_logger.log((epoch, epoch, epoch, epoch), (train_reconst, train_percept, val_reconst, val_percept))

            visdom_saver.save()

        print('*' * 65)
        print('Finish train and test on all epoch')


#------------------------------------------------------------------------------
#Testing on specific images
#------------------------------------------------------------------------------

class Tester():
    """Testing trained model on test data.
    """
    def test(self, opt, neuralnet, dataloader, source_sampler, target_sampler):
        """
        Segment on random image from dataset
        Support 2D images only
        """

        neuralnet.eval()
        idx = 0
        
        for iteration, batch in enumerate(dataloader):
            with torch.no_grad():
                x = batch[0].requires_grad_(False).to(DEVICE)
                # y = batch[1].requires_grad_(False).to(DEVICE)
                x_prime, _ = target_sampler.forward(x)
                x, _ = source_sampler.forward(x)

                output, _, gauss_mu = neuralnet(x, x_prime)

                predict = output.detach().cpu().numpy()
                gauss_mu = gauss_mu.detach().cpu().numpy()
                # gauss_map = gauss_map.detach().cpu().numpy()
                # seg = seg.max(dim=1)[1].detach().cpu().numpy()

                os.makedirs('testcheck', exist_ok=True)
                fig_path = path.join('testcheck', 'fig_{}.png'.format(iteration))
                utils.savegrid(fig_path, x_prime.cpu()[:, :1, :].numpy(), predict, gauss_mu=gauss_mu, name='deform')

                idx += x.shape[0]

        neuralnet.train()
        return idx

    def exe(self, opt):
        #Load trained model
        print(opt, '\n')
        print('Load checkpoint at {}'.format(opt.trained_model))

        neuralnet = AssembleNet()
        checkpoint = torch.load(opt.trained_model, map_location=lambda storage, loc: storage, pickle_module=pickle)
        model_state = checkpoint['modelstate']
        neuralnet.load_state_dict(model_state)

        model_parameters = neuralnet.parameters() #filter(lambda p: p.requires_grad, neuralnet.parameters())
        n_params = sum([p.numel() for p in model_parameters])
        print('number of params', n_params)

        #Dataloader
        testset = data.get_valset(opt.valset) #data.get_testset(opt.testset)
        testLoader = DataLoader(dataset=testset, num_workers=opt.nthreads, batch_size=opt.val_batch_size, shuffle=False)

        #DataParallel for multiple GPUs:
        if len(opt.gpus) > 1:
            #dim always is 0 because of input data always is in shape N*W
            neuralnet = nn.DataParallel(neuralnet, device_ids=opt.gpus, dim=0)
        neuralnet.to(DEVICE)

        print('Start testing on device {}'.format(DEVICE.type))
        start_time = time.time()
        total_sample = self.test(opt, neuralnet, testLoader, source_sampler,target_sampler)
        print('| finish testing on {} samples in {} seconds'.format(total_sample, time.time() - start_time))


if __name__ == "__main__":
    if ARGS.trained_model:
        tester = Tester()
        tester.exe(ARGS)
    else:
        main = Main(ARGS)
        main.exe()
