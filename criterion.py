import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg import Vgg16

class LossFunc(nn.Module):
    """
    Loss function for landmark prediction
    """
    def __init__(self, loss_type='perceptual'):
        super(LossFunc, self).__init__()
        self.loss_type = loss_type
        self.ema = EMA()
        self.vggnet = Vgg16() if loss_type == 'perceptual' else None

    def forward(self, future_im_pred, future_im, mask=None):
        loss = self._loss(future_im_pred, future_im, mask=mask)
        return loss

    def _loss(self, future_im_pred, future_im, mask=None):
        "loss function"
        vgg_losses = []
        w_reconstruct = 1. / 255.
        if self.loss_type == 'perceptual':
            w_reconstruct = 1.
            reconstruction_loss, vgg_losses = self._colorization_reconstruction_loss(
                future_im, future_im_pred, mask=mask)
        elif self.loss_type == 'l2':
            if mask is not None:
                l = F.mse_loss(future_im_pred, future_im, reduction='none')
                reconstruction_loss = torch.mean(self._loss_mask(l, mask))
            else:
                reconstruction_loss = F.mse_loss(future_im_pred, future_im)
        else:
            raise ValueError('Incorrect loss-type')

        loss = w_reconstruct * reconstruction_loss

        return loss, vgg_losses

    def _loss_mask(self, imap, mask):
        mask = F.interpolate(mask, imap.shape[-2:])
        return imap * mask

    def _colorization_reconstruction_loss(
        self, gt_image, pred_image, mask=None, \
        names=['input', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']):
        #init weight
        ws = [50., 40., 6., 3., 3., 1.]

        #get features map from vgg
        feats_gt = self.vggnet(gt_image)
        feats_pred = self.vggnet(pred_image)

        feat_gt, feat_pred = [gt_image], [pred_image]
        for k in names[1:]: #no need input
            feat_gt.append(getattr(feats_gt, k))
            feat_pred.append(getattr(feats_pred, k))

        losses = []
        for k, _ in enumerate(names):
            l = F.mse_loss(feat_pred[k], feat_gt[k], reduction='none')
            if mask is not None:
                l = self._loss_mask(l, mask)
            wl = self._exp_running_avg(
                torch.mean(l).item(), init_val=ws[k], name=names[k])
            l /= wl
            l = torch.mean(l)
            losses.append(l)
        vgg_losses = [x.item() for x in losses] #for display
        loss = torch.stack(losses).sum()
        return loss, vgg_losses

    def _exp_running_avg(self, x, init_val=0., name='x'):
        with torch.no_grad():
            if not self.training:
                return init_val
            x_new = self.ema.update(name, x, init_val)
            return x_new


class EMA(object):
    """Exponential running average
    """
    def __init__(self, decay=0.99):
        self.rho = decay
        self.avgs = {}

    def register(self, name, val):
        "add val to shadow by key=name"
        self.avgs.update({name: val})

    def get(self, name):
        "get value with key=name"
        return self.avgs[name]

    def update(self, name, x, init_val=0.):
        "update new value for variable x"
        if name not in self.avgs.keys():
            self.register(name, init_val)
            return init_val

        x_avg = self.get(name)
        w_update = 1. - self.rho
        x_new = x_avg + w_update * (x - x_avg)
        self.register(name, x_new)
        return x_new
