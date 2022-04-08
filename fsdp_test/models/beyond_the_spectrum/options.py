class Options:
    def __init__(self, name):
        assert name == 'pixel' or name == 'stage5'

        setattr(self, 'arch', 'resnet50')
        setattr(self, 'scale_factor', 4)
        setattr(self, 'num_features', 64)
        setattr(self, 'growth_rate', 64)
        setattr(self, 'num_blocks', 16)
        setattr(self, 'num_layers', 8)
        if name == 'pixel':
            setattr(self, 'input_channel', 3)
            setattr(self, 'idx_stages', 0)
            setattr(self, 'weights_cls_path', './models/beyond_the_spectrum/weights/pixel_pggan_celeba.pth.tar')
            setattr(self, 'weights_sr_path', './models/beyond_the_spectrum/weights/sr_vgg_epoch_last.pth')
        elif name == 'stage5':
            setattr(self, 'input_channel', 512)
            setattr(self, 'idx_stages', 5)
            setattr(self, 'weights_cls_path', './models/beyond_the_spectrum/weights/stylegan_celeba_stage5_noising/cls.pth.tar')
            setattr(self, 'weights_sr_path', './models/beyond_the_spectrum/weights/stylegan_celeba_stage5_noising/sr.pth.tar')