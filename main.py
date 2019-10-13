"""
Train and Evaluate Model
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
import imm_model as model
from imm_model import AssembleNet
from criterion import *

parser = argparse.ArgumentParser(description='Option for Glaucoma')
#------------------------------------------------------------------- data-option
parser.add_argument('--data_root', type=str,
                    default='../data/',
                    help='location of root dir')
parser.add_argument('--dataset', type=str,
                    default='celeba',
                    help='location of dataset')
parser.add_argument('--testset', type=str,
                    default='../data/',
                    help='location of test data')
parser.add_argument('--nthreads', type=int, default=8,
                    help='number of threads for data loader')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='train batch size')
parser.add_argument('--val_batch_size', type=int, default=8, metavar='N',
                    help='val batch size')
#------------------------------------------------------------------ model-option
parser.add_argument('--pretrained_model', type=str, default='',
                    help='pretrain model location')
#--------------------------------------------------------------- training-option
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--gpus', type=list, default=[2, 3],
                    help='list of GPUs in use')
#optimizer-option
parser.add_argument('--optim_algor', type=str, default='Adam',
                    help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-8,
                    help='weight_decay rate')
#saving-option
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
parser.add_argument('--checkpoint_interval', type=int, default=1,
                    help='epoch interval of saving checkpoint')
parser.add_argument('--save_path', type=str, default='checkpoint',
                    help='directory for saving checkpoint')
parser.add_argument('--resume_checkpoint', type=str, default='',
                    help='location of saved checkpoint')
#only prediction-option
parser.add_argument('--trained_model', type=str, default='',
                    help='location of trained checkpoint')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpus[0]) if len(args.gpus) > 0 else 'cpu')
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if device.type == 'cuda':
    cuda.set_device(args.gpus[0])
    cuda.manual_seed(args.seed)

class Main:
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(self.opt.save_path, exist_ok=True)

        self.neuralnet = self._make_model(opt)
        self.optimizer = self._make_optimizer(opt, self.neuralnet)
        self.train_loader, self.val_loader = self._make_data(opt)

        #loss function
        self.criterion = nn.MSELoss()

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
                im = batch[0].requires_grad_(False).to(device)
                keypts = batch[1].requires_grad_(False).to(device)

                im, future_im, mask, _, _ = self.batch_transform.exe(im, landmarks=keypts)

                future_im_pred, gauss_yx, pose_embeddings = self.neuralnet(im, future_im)

                #vgg loss

                #log meter
                self.loss_meter.add(loss.item())
                # self.score_meter.add(jaccscore.item())

                # if epoch % 50 == 1 and iteration == 0:
                #     shape = target.shape
                #     predict = output.reshape(shape[0], shape[2], shape[3], shape[1]).max(dim=3)[1] #dim 3 now is classes
                #     os.makedirs('valcheck', exist_ok=True)
                #     fig_path = path.join('valcheck', 'fig_{}.png'.format(epoch))
                #     savegrid(fig_path, input.cpu()[:, :1, :].numpy(), predict.cpu().numpy(), labels=label.cpu().numpy())

        self.neuralnet.train()
        return self.loss_meter.value()[0]

    def _train(self, dataloader, epoch):
        self.neuralnet.train()
        self._resetmeter()

        for iteration, batch in enumerate(dataloader, 1):
            start_time = time.time()

            im = batch[0].requires_grad_(False).to(device)
            keypts = batch[1].requires_grad_(False).to(device)

            im, future_im, mask, _, _ = self.batch_transform.exe(im, landmarks=keypts)

            future_im_pred, gauss_yx, pose_embeddings = self.neuralnet(im, future_im)

            #vgg loss


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

    def execute(self):
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

        self.neuralnet.to(device)
        self.vgg.to(device)

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
            self.opt.optim_algor, device.type))

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

class Tester:
    def test(self, opt, neuralnet, dataloader, source_sampler, target_sampler):
        """
        Segment on random image from dataset
        Support 2D images only
        """

        neuralnet.eval()
        idx = 0
        
        for iteration, batch in enumerate(dataloader):
            with torch.no_grad():
                x = batch[0].requires_grad_(False).to(device)
                # y = batch[1].requires_grad_(False).to(device)
                x_prime, _ = target_sampler.forward(x)
                x, _ = source_sampler.forward(x)

                output, _, gauss_mu = neuralnet(x, x_prime)

                predict = output.detach().cpu().numpy()
                gauss_mu = gauss_mu.detach().cpu().numpy()
                # gauss_map = gauss_map.detach().cpu().numpy()
                # seg = seg.max(dim=1)[1].detach().cpu().numpy()

                os.makedirs('testcheck', exist_ok=True)
                fig_path = path.join('testcheck', 'fig_{}.png'.format(iteration))
                savegrid(fig_path, x_prime.cpu()[:, :1, :].numpy(), predict, gauss_mu=gauss_mu, name='deform')

                # fig_path_seg = path.join('testcheck', 'figseg_{}.png'.format(iteration))
                # savegrid(fig_path_seg, x.cpu()[:, :1, :].numpy(), seg, name='img')

                # os.makedirs('submit', exist_ok=True)
                # assert len(filenames) == predict.shape[0], '***ERROR: mismatch batch size***'
                # predict = predict.permute(0, 2, 3, 1).cpu() #B*H*W*C
                # for i in range(predict.shape[0]):
                #     f_path = path.join('testcheck', '{}.npy'.format(filenames[i]))
                #     f_data = predict[i].numpy().astype(np.uint8)
                #     try:
                #         np.save(f_path, f_data)
                #     except:
                #         print('*** ALERT: ERROR AT {} ***'.format(f_path))

                idx += x.shape[0]

        neuralnet.train()
        return idx

    def execute(self, opt):
        #Load trained model
        print(opt, '\n')
        print('Load checkpoint at {}'.format(opt.trained_model))

        neuralnet = AssembleNet()
        checkpoint = torch.load(opt.trained_model, map_location=lambda storage, loc: storage, pickle_module=pickle)
        model_state = checkpoint['modelstate']
        neuralnet.load_state_dict(model_state)

        source_sampler = TPSRandomSampler(height=224, width=224,\
            rotsd=5., scalesd=0.1, warpsd=[0.001, 0.01], transsd=0.1, pad=False)
        target_sampler = TPSRandomSampler(height=224, width=224,\
            rotsd=0., scalesd=0., warpsd=[0.001, 0.005], transsd=0.1, pad=False)

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
        neuralnet.to(device)

        print('Start testing on device {}'.format(device.type))
        start_time = time.time()
        total_sample = self.test(opt, neuralnet, testLoader, source_sampler,target_sampler)
        print('| finish testing on {} samples in {} seconds'.format(total_sample, time.time() - start_time))

def savegrid(fig_path, images, predictions, gauss_mu=None, labels=None, nrow=8, ncol=8, name='image'):
    step = 2
    ncol = 8
    fig_width = 20
    if labels is not None:
        step = 3
        ncol = 12
        fig_width = 30
    plt.rcParams['figure.figsize'] = (fig_width, 40)
    j = 0
    for i in range(0, nrow*ncol, step):
        if j >= len(images):
            break
        img = images[j].squeeze()
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img) #,interpolation='none', cmap="nipy_spectral")
        if gauss_mu is not None:
            for k in range(gauss_mu[j].shape[0]):
                y_jk = ((gauss_mu[j, k, 0]+1)*15*16).astype(np.int)
                x_jk = ((gauss_mu[j, k, 1]+1)*11*16).astype(np.int)
                plt.plot(x_jk, y_jk, 'bo')
        plt.title('{}_{}'.format(name, j))
        plt.axis('off')

        pred = predictions[j].squeeze()
        plt.subplot(nrow, ncol, i+2)
        plt.imshow(pred)
        plt.title('predict_{}'.format(j))
        plt.axis('off')

        if labels is not None:
            label = labels[j]
            plt.subplot(nrow, ncol, i+3)
            plt.imshow(label)
            plt.title('label_{}'.format(j))
            plt.axis('off')

        j += 1
    # plt.show()
    plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def transform_image(img):
    " randomly transform input images"
    #random vertical flip -- axis 1
    # if np.random.uniform() > 0.5:
        # img = torch.flip(img, dims=[2,])
    #elastic transform
    indices_x_clipped, indices_y_clipped = data.create_elastic_indices()
    img[:, :, :, :] = img[:, :, indices_y_clipped, indices_x_clipped]
    return img


if __name__ == "__main__":
    if opt.trained_model:
        tester = Tester()
        tester.execute(opt)
    else:
        main = Main(opt)
        main.execute()
