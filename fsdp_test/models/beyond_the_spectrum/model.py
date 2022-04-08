import torch
from torch import nn

from .resnet import get_resnet
from .sr_model import ModelParallelRDN, RDN, Vgg19

class BeyondtheSpectrumModelParallel(nn.Module):
    def __init__(self, opt):
        super(BeyondtheSpectrumModelParallel, self).__init__()

        self.opt = opt
        
        self.sr_model = ModelParallelRDN(scale_factor=opt.scale_factor,
                                            num_channels=3,
                                            num_features=opt.num_features,
                                            growth_rate=opt.growth_rate,
                                            num_blocks=opt.num_blocks,
                                            num_layers=opt.num_layers,
                                            requires_grad=False)
        self.perception_model = Vgg19().to('cuda:3')
        self.cls_model = get_resnet(arch=opt.arch, 
                                    pretrained = False,
                                    input_channel=opt.input_channel,
                                    num_classes=2, 
                                    dilated=False).to('cuda:3')


    def load_weights(self):
        cls_weights = torch.load(self.opt.weights_cls_path, map_location='cpu')
        sr_weights = torch.load(self.opt.weights_sr_path, map_location='cpu')

        self.cls_model.load_state_dict(cls_weights['state_dict'], strict=False)
        if 'state_dict' in sr_weights.keys():
            self.sr_model.load_state_dict(sr_weights['state_dict'])
        else:
            self.sr_model.load_state_dict(sr_weights)

    
    def scale_input_func(self, x, scale_factor):
        lr = 0
        for ii in range(scale_factor):
            for jj in range(scale_factor):
                lr = lr + x[:, :, ii::scale_factor, jj::scale_factor] / (scale_factor * scale_factor)

        lr = lr / 255.0

        return lr


    def forward(self, x):
        lr = self.scale_input_func(x, self.opt.scale_factor)
        x = x / 255.0
        preds_input = self.sr_model(lr)

        if self.opt.idx_stages > 0:
            per_rec = self.perception_model(preds_input.to('cuda:3'))
            per_gt = self.perception_model(x.to('cuda:3'))

            rec_features = abs(per_rec[self.opt.idx_stages-1]-per_gt[self.opt.idx_stages-1])
        else:
            rec_features = abs(preds_input.to('cuda:3')-x.to('cuda:3'))

        output = self.cls_model(rec_features)

        return output


class BeyondtheSpectrumModel(nn.Module):
    def __init__(self, opt):
        super(BeyondtheSpectrumModel, self).__init__()

        self.opt = opt

        self.sr_model = RDN(scale_factor=opt.scale_factor,
                            num_channels=3,
                            num_features=opt.num_features,
                            growth_rate=opt.growth_rate,
                            num_blocks=opt.num_blocks,
                            num_layers=opt.num_layers,
                            requires_grad=False)
        self.perception_model = Vgg19()
        self.cls_model = get_resnet(arch=opt.arch, 
                                    pretrained = False,
                                    input_channel=opt.input_channel,
                                    num_classes=2, 
                                    dilated=False)


    def load_weights(self):
        cls_weights = torch.load(self.opt.weights_cls_path, map_location='cpu')
        sr_weights = torch.load(self.opt.weights_sr_path, map_location='cpu')

        self.cls_model.load_state_dict(cls_weights['state_dict'], strict=False)
        if 'state_dict' in sr_weights.keys():
            self.sr_model.load_state_dict(sr_weights['state_dict'])
        else:
            self.sr_model.load_state_dict(sr_weights)

    
    def scale_input_func(self, x, scale_factor):
        lr = 0
        for ii in range(scale_factor):
            for jj in range(scale_factor):
                lr = lr + x[:, :, ii::scale_factor, jj::scale_factor] / (scale_factor * scale_factor)

        lr = lr / 255.0

        return lr


    def forward(self, x):
        lr = self.scale_input_func(x, self.opt.scale_factor)
        x = x / 255.0
        preds_input = self.sr_model(lr)

        if self.opt.idx_stages > 0:
            per_rec = self.perception_model(preds_input)
            per_gt = self.perception_model(x)

            rec_features = abs(per_rec[self.opt.idx_stages-1]-per_gt[self.opt.idx_stages-1])
        else:
            rec_features = abs(preds_input-x)

        output = self.cls_model(rec_features)

        return output