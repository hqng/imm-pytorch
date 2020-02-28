"""
Train Evaluate and Test Model
"""

import os, argparse, gc, glob, time, pickle
from os import path

import torch
from torch import nn, optim, cuda
from torch.utils.data import DataLoader
from visdom import Visdom
from torchnet import meter
from torchnet.logger import VisdomPlotLogger, VisdomSaver

import data
import utils
from imm_model import AssembleNet
from criterion import LossFunc


PARSER = argparse.ArgumentParser(description='Option for Conditional Image Generating')
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
PARSER.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='train batch size')
PARSER.add_argument('--val_batch_size', type=int, default=64, metavar='N',
                    help='val batch size')
#------------------------------------------------------------------ model-option
PARSER.add_argument('--pretrained_model', type=str, default='',
                    help='pretrain model location')
PARSER.add_argument('--loss_type', type=str, default='perceptual',
                    help='loss type for criterion: perceptual | l2')
#--------------------------------------------------------------- training-option
PARSER.add_argument('--seed', type=int, default=1234,
                    help='random seed')
PARSER.add_argument('--gpus', type=list, default=[3],
                    help='list of GPUs in use')
#optimizer-option
PARSER.add_argument('--optim_algor', type=str, default='Adam',
                    help='optimization algorithm')
PARSER.add_argument('--lr', type=float, default=1e-3,
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


def _make_model(opt):
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


def _make_optimizer(opt, neuralnet, param_groups=None):
    parameters = filter(lambda p: p.requires_grad, neuralnet.parameters())

    if param_groups is not None:
        lr = param_groups[0]['lr']
        weight_decay = param_groups[0]['weight_decay']
    else:
        lr = opt.lr
        weight_decay = opt.weight_decay

    optimizer = getattr(optim, opt.optim_algor)(
        parameters, lr=lr, weight_decay=weight_decay)

    return optimizer


def _make_data(opt, subset='train', shuffle=True):
    #get data
    split = data.get_dataset(opt.data_root, opt.dataset, subset=subset)

    #dataloader
    loader = DataLoader(dataset=split, \
        num_workers=opt.nthreads, batch_size=opt.batch_size, shuffle=shuffle)

    return loader


class Main:
    """Wrap training and evaluating processes
    """
    def __init__(self, opt):
        self.opt = opt
        os.makedirs(self.opt.save_path, exist_ok=True)

        self.neuralnet = _make_model(opt)
        self.optimizer = _make_optimizer(opt, self.neuralnet)
        self.train_loader = _make_data(opt)
        self.val_loader = _make_data(opt, subset='val', shuffle=False)

        #loss function
        self.criterion = LossFunc(opt.loss_type)

        #batch data transform
        self.batch_transform = data.BatchTransform()

        #meter
        self.loss_meter = meter.AverageValueMeter()


    #===========================================================================
    # Training and Evaluating
    #===========================================================================

    def _resetmeter(self):
        self.loss_meter.reset()

    def _evaluate(self, dataloader):
        gc.collect()
        self._resetmeter()

        self.neuralnet.eval()
        self.criterion.eval()

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                im = batch[0].requires_grad_(False).to(DEVICE)
                keypts = batch[1].requires_grad_(False).to(DEVICE)

                deformed_batch = self.batch_transform.exe(im, landmarks=keypts)
                im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask']

                future_im_pred, _, _ = self.neuralnet(im, future_im)

                #loss
                loss, _ = self.criterion(future_im_pred, future_im)

                #log meter
                self.loss_meter.add(loss.item())

        self.neuralnet.train()
        self.criterion.train()

        return self.loss_meter.value()[0]

    def _train(self, dataloader, epoch):
        self._resetmeter()

        self.neuralnet.train()
        self.criterion.train()

        for iteration, batch in enumerate(dataloader, 1):
            start_time = time.time()

            im = batch[0].to(DEVICE)
            keypts = batch[1].to(DEVICE)

            deformed_batch = self.batch_transform.exe(im, landmarks=keypts)
            im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask']

            #zero gradient first,then forward
            self.optimizer.zero_grad()
            future_im_pred, _, _ = self.neuralnet(im, future_im)

            #loss
            loss, loss_values = self.criterion(future_im_pred, future_im)

            loss.backward()
            self.optimizer.step()

            #update weight of perceptual loss by EMA
            for k, new_val in enumerate(loss_values):
                tmp_name = list(self.criterion.ema.avgs)[k]
                tmp_init_val = self.criterion.ema.avgs[tmp_name]
                self.criterion.ema.update(tmp_name, new_val, init_val=tmp_init_val)

            #log meter
            self.loss_meter.add(loss.item())

            #print
            eslapsed = time.time() - start_time
            print('| epoch {:3d} | {:3d}/{:3d} ith_batch | time(s) {:5.2f} | \n loss {:5.2f} | vgg losses {} \n'.format( \
                    epoch, iteration, len(dataloader), eslapsed, loss.item(), loss_values))

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
            self.optimizer = _make_optimizer(
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
        viz = Visdom(port=port)
        visdom_saver = VisdomSaver([viz.env])

        loss_logger = VisdomPlotLogger('line', port=port, \
            opts={'title': 'Total Loss', 'legend': ['train', 'val']})

        print('Start training: optim {}, on device {}'.format( \
            self.opt.optim_algor, DEVICE))

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=5e-5)

        for epoch in range(start_epoch, self.opt.epochs+1):
            #let's go
            print('\n')
            print('-' * 65)
            print('{}'.format(time.asctime(time.localtime())))
            print(' **Training epoch {}, lr {}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            start_time = time.time()
            train_loss = self._train(self.train_loader, epoch)

            print('| finish training on epoch {:3d} | time(s) {:5.2f} | loss {:3.4f}'.format(
                epoch, time.time() - start_time, train_loss))

            print(' **Evaluating on validate set')

            start_time = time.time()
            val_loss = self._evaluate(self.val_loader)

            print('| finish validating on epoch {:3d} | time(s) {:5.2f} | loss {:3.4f}'.format(
                epoch, time.time() - start_time, val_loss))

            #save check point
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
            visdom_saver.save()

            #update learning rate
            lr_scheduler.step()

        print('*' * 65)
        print('Finish train and test all epochs')


#------------------------------------------------------------------------------
#Testing on specific images
#------------------------------------------------------------------------------

class Tester():
    """Testing trained model on test data.
    """
    @staticmethod
    def test(neuralnet, dataloader):
        """
        Segment on random image from dataset
        Support 2D images only
        """
        neuralnet.eval()
        batch_transform = data.BatchTransform()

        idx = 0
        for iteration, batch in enumerate(dataloader):
            with torch.no_grad():
                im = batch[0].requires_grad_(False).to(DEVICE)
                keypts = batch[1].requires_grad_(False).to(DEVICE)

                deformed_batch = batch_transform.exe(im, landmarks=keypts)
                im, future_im, mask = deformed_batch['image'], deformed_batch['future_image'], deformed_batch['mask']

                future_im_pred, gauss_mu, _ = neuralnet(im, future_im)

                predict = future_im_pred.data.cpu().numpy().transpose(0, 2, 3, 1)
                gauss_mu = gauss_mu.data.cpu().numpy()
                # gauss_map = gauss_map.data.cpu().numpy()
                future_im = future_im.data.cpu().numpy().transpose(0, 2, 3, 1)

                os.makedirs('testcheck', exist_ok=True)
                fig_path = path.join('testcheck', 'fig_{}.png'.format(iteration))
                utils.savegrid(fig_path, future_im, predict, gauss_mu=gauss_mu, name='deform')

                idx += im.shape[0]

        neuralnet.train()
        return idx

    @staticmethod
    def exe(opt):
        #Load trained model
        print(opt, '\n')
        print('Load checkpoint at {}'.format(opt.trained_model))

        neuralnet = _make_model(opt)
        checkpoint = torch.load(opt.trained_model, \
            map_location=lambda storage, loc: storage, pickle_module=pickle)
        model_state = checkpoint['modelstate']
        neuralnet.load_state_dict(model_state)

        #Dataloader
        test_loader = _make_data(opt, subset='val', shuffle=False)

        #DataParallel for multiple GPUs:
        if len(opt.gpus) > 1:
            #dim always is 0 because of input data always is in shape N*W
            neuralnet = nn.DataParallel(neuralnet, device_ids=opt.gpus, dim=0)
        neuralnet.to(DEVICE)

        print('Start testing on device {}'.format(DEVICE.type))
        start_time = time.time()
        total_sample = Tester.test(neuralnet, test_loader)
        print('| finish testing on {} samples in {} seconds'.format(
            total_sample, time.time() - start_time))

if __name__ == "__main__":
    if ARGS.trained_model:
        Tester.exe(ARGS)
    else:
        main = Main(ARGS)
        main.exe()
