import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.facenet import Sphere20a, senet50

import intrinsic_flow_models.networks as ifm_networks


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], model0_res=0, model1_res=0, extra_channel=3, div=3, disp=1, regarch=4):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_style2_9blocks':
        net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=model0_res, extra_channel=extra_channel)
    elif netG == 'resnet_9blocks_rcatland':
        net = ResnetConditionTriGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_rcatland3':
        net = ResnetConditionTriGenerator3(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32':
        net = ResnetConditionTriGenerator32(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_10blocks_rcatland32':
        net = ResnetConditionTriGenerator32(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=10, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland4':
        net = ResnetConditionTriGenerator4(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_rcatland32_fw':
        net = ResnetConditionTriGenerator32_fw(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_fw2':
        net = ResnetConditionTriGenerator32_fw2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_ifw':
        net = ResnetConditionTriGenerator32_ifw(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_ifw_single2':
        net = ResnetConditionTriGenerator32_ifw_single2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw':
        net = ResnetConditionTriGenerator32_full_ifw(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw_colorcoded':
        net = ResnetConditionTriGenerator32_full_ifw_colorcoded(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw2':
        net = ResnetConditionTriGenerator32_full_ifw2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw_single':
        net = ResnetConditionTriGenerator32_full_ifw_single(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw_single2':
        net = ResnetConditionTriGenerator32_full_ifw_single2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'resnet_9blocks_rcatland32_full_ifw_single3':
        net = ResnetConditionTriGenerator32_full_ifw_single(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, div=div, disp=disp)
    elif netG == 'regressor':
        net = Regressor(input_nc, ngf, norm_layer=norm_layer, arch=regarch)
    elif netG == 'combiner':
        net = Combiner(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=2)
    elif netG == 'resnet_9blocks_rcatland2':
        net = ResnetConditionTriGenerator2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], n_class=3):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_cls':
        net = NLayerDiscriminatorCls(input_nc, ndf, n_layers=3, n_class=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_ULP(landmark_num=10, init_type='normal', init_gain=0.02, gpu_ids=[], init_weights_ = None):
    net = ULP(landmark_num)
    net = init_net(net, init_type, init_gain, gpu_ids)

    if not init_weights_ == None:
        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        print('Loading model from: %s'%init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(device))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print('load the weights successfully')
    
    return net

def define_HED(init_weights_, gpu_ids_=[]):
    net = HED()

    if len(gpu_ids_) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids_[0])
        net = torch.nn.DataParallel(net, gpu_ids_)  # multi-GPUs
    
    if not init_weights_ == None:
        device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
        print('Loading model from: %s'%init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(device))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print('load the weights successfully')

    return net

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_P(segment_classes, input_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netP = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'unet_128':
        netP = UnetParseGenerator(segment_classes, input_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netP = UnetParseGenerator(segment_classes, input_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    #print(netP)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netP.cuda(gpu_ids[0])
    netP.apply(weights_init)
    return netP

class UnetParseGenerator(nn.Module):

    def __init__(self, segment_classes, input_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetParseGenerator, self).__init__()
        output_nc = segment_classes
        # construct unet structure
        unet_block = UnetParseSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetParseSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetParseSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetParseSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetParseSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetParseSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        #maybe do some check here with softmax
        self.model = unet_block

    def forward(self, input):
        softmax = torch.nn.Softmax(dim = 1)
        return softmax(self.model(input))

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetParseSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetParseSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        global printlayer_index
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1,output_padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # printlayer = [PrintLayer(name = str(printlayer_index))]
            # printlayer_index += 1
            # model = printlayer + down + [submodule] + up
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = printlayer + down + up
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
                # model = printlayer + down + [submodule] + printlayer + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule]  + up
                # model = printlayer + down + [submodule] + printlayer + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        model_output = self.model(x)
        wb,hb = model_output.size()[3],model_output.size()[2]
        wa,ha = x.size()[3],x.size()[2]
        l = int((wb-wa)/2)
        t = int((hb-ha)/2)
        model_output = model_output[:,:,t:t+ha,l:l+wa]
        if self.outermost:
            return model_output
        else:
            return torch.cat([x, model_output], 1)           #if not the outermost block, we concate x and self.model(x) during forward to implement unet

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':#cyclegan
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetStyle2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', extra_channel=3, model0_res=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetStyle2Generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        mult = 2 ** n_downsampling
        for i in range(model0_res):       # add ResNet blocks
            model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model = []
        model += [nn.Conv2d(ngf * mult + extra_channel, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        for i in range(n_blocks-model0_res):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        self.model = nn.Sequential(*model)
        #print(list(self.modules()))

    def forward(self, input1, input2):
        """Standard forward"""
        f1 = self.model0(input1)
        y1 = torch.cat([f1, input2], 1)
        return self.model(y1)

class ResnetConditionTriGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = 3
        super(ResnetConditionTriGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + 1) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock2(ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)

    def forward(self, input, land1, land2):
        """Standard forward"""
        x = self.model1(input)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + 1) % self.div == 0:
                x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out

class ResnetConditionTriGenerator3(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator3, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))

    def forward(self, input, land1, land2):
        """Standard forward"""
        x = self.model1(input)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out

class ResnetConditionTriGenerator32(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))

    def forward(self, input, land1, land2):
        """Standard forward"""
        x = self.model1(input)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_fw(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_fw, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))

    def forward(self, input, land1, land2, motion):
        """Standard forward"""
        motion64 = motion.permute(0, 3, 1, 2)
        motion64 = F.interpolate(motion64, size=(64, 64), mode='bilinear', align_corners=True)
        motion64 = motion64.permute(0, 2, 3, 1)
        x = self.model1(input)
        x = F.grid_sample(x, motion64)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                #x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_fw2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_fw2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        model1 = []
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model0 = nn.Sequential(*model0)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))

    def forward(self, input, land1, land2, motion):
        """Standard forward"""
        x = self.model0(input)
        x = F.grid_sample(x, motion)
        #print(x.shape)
        x = self.model1(x)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                #x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_ifw(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_ifw, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        bgf_half = int(ngf/2)
        model0 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, bgf_half, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(bgf_half),
                 nn.ReLU(True)]
        
        model1 = []
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model0 = nn.Sequential(*model0)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x = self.model0(input)
        x1 = F.grid_sample(x, motion)
        x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        #print(x1.shape, x2.shape)
        x = self.model1(torch.cat([x1, x2], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                #x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_full_ifw(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_full_ifw, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, int(ngf/2), kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(int(ngf/2)),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri10 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri11 = [nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1),
                  nn.ReLU(True)]       #here
        model_tri12 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri20 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri21 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri22 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]#here
                  
        self.model_tri_merge = nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, padding=1, bias=use_bias)
        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model_tri10 = nn.Sequential(*model_tri10)
        self.model_tri11 = nn.Sequential(*model_tri11)
        self.model_tri12 = nn.Sequential(*model_tri12)
        self.model_tri20 = nn.Sequential(*model_tri20)
        self.model_tri21 = nn.Sequential(*model_tri21)
        self.model_tri22 = nn.Sequential(*model_tri22)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        #print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, flow, ifmask, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/2, size=(128, 128), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(128, 128), mode='bilinear', align_corners=True)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/4, size=(64, 64), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(64, 64), mode='bilinear', align_corners=True)
        x1 = F.grid_sample(x, motion)
        x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        return torch.cat([x1, x2], 1)

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, flow, ifmask, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        x2 = self.model_tri10(input)
        x2 = self.model_tri11(x2)
        x2 = self.double_feature_warping(x2, motion, flow, ifmask, 1)
        x2 = self.model_tri12(x2)
        x3 = self.model_tri20(input)
        x3 = self.model_tri21(x3)
        x3 = self.model_tri22(x3)
        x3 = self.double_feature_warping(x3, motion, flow, ifmask, 2)

        x = self.model_tri_merge(torch.cat([x1, x2, x3], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out

class ResnetConditionTriGenerator32_full_ifw_colorcoded(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_full_ifw_colorcoded, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, int(ngf/2), kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(int(ngf/2)),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri10 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri11 = [nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1),
                  nn.ReLU(True)]       #here
        model_tri12 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri20 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri21 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri22 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]#here
                  
        self.model_tri_merge = nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, padding=1, bias=use_bias)
        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model_tri10 = nn.Sequential(*model_tri10)
        self.model_tri11 = nn.Sequential(*model_tri11)
        self.model_tri12 = nn.Sequential(*model_tri12)
        self.model_tri20 = nn.Sequential(*model_tri20)
        self.model_tri21 = nn.Sequential(*model_tri21)
        self.model_tri22 = nn.Sequential(*model_tri22)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, flow, ifmask, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/2, size=(128, 128), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(128, 128), mode='bilinear', align_corners=True)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/4, size=(64, 64), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(64, 64), mode='bilinear', align_corners=True)
        x1 = F.grid_sample(x, motion)
        x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        return torch.cat([x1, x2], 1)

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, flow, ifmask, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        x2 = self.model_tri10(input)
        x2 = self.model_tri11(x2)
        x2 = self.double_feature_warping(x2, motion, flow, ifmask, 1)
        x2 = self.model_tri12(x2)
        x3 = self.model_tri20(input)
        x3 = self.model_tri21(x3)
        x3 = self.model_tri22(x3)
        x3 = self.double_feature_warping(x3, motion, flow, ifmask, 2)

        x = self.model_tri_merge(torch.cat([x1, x2, x3], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
    
class ResnetConditionTriGenerator32_full_ifw2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_full_ifw2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, int(ngf/2), kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(int(ngf/2)),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri10 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri11 = [nn.Conv2d(ngf * 1, ngf * 1, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1),
                  nn.ReLU(True)]       #here
        model_tri12 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri20 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri21 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri22 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]#here
                  
        self.model_tri_merge = nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, padding=1, bias=use_bias)
        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model_tri10 = nn.Sequential(*model_tri10)
        self.model_tri11 = nn.Sequential(*model_tri11)
        self.model_tri12 = nn.Sequential(*model_tri12)
        self.model_tri20 = nn.Sequential(*model_tri20)
        self.model_tri21 = nn.Sequential(*model_tri21)
        self.model_tri22 = nn.Sequential(*model_tri22)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, motion2, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            motion2 = motion2.permute(0, 3, 1, 2)
            motion2 = F.interpolate(motion2, size=(128, 128), mode='bilinear', align_corners=True)
            motion2 = motion2.permute(0, 2, 3, 1)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            motion2 = motion2.permute(0, 3, 1, 2)
            motion2 = F.interpolate(motion2, size=(64, 64), mode='bilinear', align_corners=True)
            motion2 = motion2.permute(0, 2, 3, 1)
        x1 = F.grid_sample(x, motion)
        x2 = F.grid_sample(x, motion2)
        return torch.cat([x1, x2], 1)

    def forward(self, input, land1, land2, motion, motion2):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, motion2, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        x2 = self.model_tri10(input)
        x2 = self.model_tri11(x2)
        x2 = self.double_feature_warping(x2, motion, motion2, 1)
        x2 = self.model_tri12(x2)
        x3 = self.model_tri20(input)
        x3 = self.model_tri21(x3)
        x3 = self.model_tri22(x3)
        x3 = self.double_feature_warping(x3, motion, motion2, 2)

        x = self.model_tri_merge(torch.cat([x1, x2, x3], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_full_ifw_single(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_full_ifw_single, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri10 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri11 = [nn.Conv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]       #here
        model_tri12 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri20 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri21 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri22 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]#here
                  
        self.model_tri_merge = nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, padding=1, bias=use_bias)
        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model_tri10 = nn.Sequential(*model_tri10)
        self.model_tri11 = nn.Sequential(*model_tri11)
        self.model_tri12 = nn.Sequential(*model_tri12)
        self.model_tri20 = nn.Sequential(*model_tri20)
        self.model_tri21 = nn.Sequential(*model_tri21)
        self.model_tri22 = nn.Sequential(*model_tri22)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, flow, ifmask, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/2, size=(128, 128), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(128, 128), mode='bilinear', align_corners=True)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/4, size=(64, 64), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(64, 64), mode='bilinear', align_corners=True)
        x1 = F.grid_sample(x, motion)
        #x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        return x1

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, flow, ifmask, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        x2 = self.model_tri10(input)
        x2 = self.model_tri11(x2)
        x2 = self.double_feature_warping(x2, motion, flow, ifmask, 1)
        x2 = self.model_tri12(x2)
        x3 = self.model_tri20(input)
        x3 = self.model_tri21(x3)
        x3 = self.model_tri22(x3)
        x3 = self.double_feature_warping(x3, motion, flow, ifmask, 2)

        x = self.model_tri_merge(torch.cat([x1, x2, x3], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
class ResnetConditionTriGenerator32_full_ifw_single2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_full_ifw_single2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri10 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri11 = [nn.Conv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.ReLU(True)]       #here
        model_tri12 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  
        model_tri20 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model_tri21 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri22 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]#here
                  
        self.model_tri_merge = nn.Conv2d(ngf * 12, ngf * 4, kernel_size=3, padding=1, bias=use_bias)
        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model_tri10 = nn.Sequential(*model_tri10)
        self.model_tri11 = nn.Sequential(*model_tri11)
        self.model_tri12 = nn.Sequential(*model_tri12)
        self.model_tri20 = nn.Sequential(*model_tri20)
        self.model_tri21 = nn.Sequential(*model_tri21)
        self.model_tri22 = nn.Sequential(*model_tri22)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, flow, ifmask, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/2, size=(128, 128), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(128, 128), mode='bilinear', align_corners=True)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/4, size=(64, 64), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(64, 64), mode='bilinear', align_corners=True)
        #x1 = F.grid_sample(x, motion)
        x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        return x2

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, flow, ifmask, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        x2 = self.model_tri10(input)
        x2 = self.model_tri11(x2)
        x2 = self.double_feature_warping(x2, motion, flow, ifmask, 1)
        x2 = self.model_tri12(x2)
        x3 = self.model_tri20(input)
        x3 = self.model_tri21(x3)
        x3 = self.model_tri22(x3)
        x3 = self.double_feature_warping(x3, motion, flow, ifmask, 2)

        x = self.model_tri_merge(torch.cat([x1, x2, x3], 1))
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out
        
        
class ResnetConditionTriGenerator32_ifw_single2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', div=3, disp=1):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = div
        self.disp = disp
        super(ResnetConditionTriGenerator32_ifw_single2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_tri00 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]#here
        model_tri01 = [nn.Conv2d(ngf * 1, ngf * 1 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 1 * 2),
                  nn.ReLU(True)]       
        model_tri02 = [nn.Conv2d(ngf * 2, ngf * 2 * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * 2 * 2),
                  nn.ReLU(True)]
                  

        

        n_downsampling = 2
        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + self.disp) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]
        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model_tri00 = nn.Sequential(*model_tri00)
        self.model_tri01 = nn.Sequential(*model_tri01)
        self.model_tri02 = nn.Sequential(*model_tri02)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)
        print(list(self.modules()))
        
    def double_feature_warping(self, x, motion, flow, ifmask, level):
        if level == 1:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(128, 128), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/2, size=(128, 128), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(128, 128), mode='bilinear', align_corners=True)
        if level == 2:
            motion = motion.permute(0, 3, 1, 2)
            motion = F.interpolate(motion, size=(64, 64), mode='bilinear', align_corners=True)
            motion = motion.permute(0, 2, 3, 1)
            flow = F.interpolate(flow/4, size=(64, 64), mode='bilinear', align_corners=True)
            ifmask = F.interpolate(ifmask, size=(64, 64), mode='bilinear', align_corners=True)
        #x1 = F.grid_sample(x, motion)
        x2 = ifm_networks.warp_acc_flow(x, flow, mask = ifmask)
        return x2

    def forward(self, input, land1, land2, motion, flow, ifmask):
        """Standard forward"""
        x1 = self.model_tri00(input)
        x1 = self.double_feature_warping(x1, motion, flow, ifmask, 0)
        x1 = self.model_tri01(x1)
        x1 = self.model_tri02(x1)
        

        x = x1
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        for i in range(self.n_blocks):
            if (i + self.disp) % self.div == 0:
                x = self.model2[i](torch.cat([x, l1, l2], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out

class ResnetConditionTriGenerator4(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.n_blocks2 = 6
        self.div = 3
        super(ResnetConditionTriGenerator4, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        con_dim = 16
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if (i + 1) % self.div == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]

        model2l = []
        mult = 2 ** n_downsampling
        for i in range(self.n_blocks2):       # add ResNet blocks
            if (i + 1) % self.div == 0:
                model2l += [
                    ResnetBlock2(ngf * mult + con_dim * 2, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2l += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3l = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3l += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3l += [nn.ReflectionPad2d(3)]
        model3l += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3l += [nn.Tanh()]

        model_landmark_trans = [nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(8), nn.ReLU(True),
                                nn.Conv2d(8, con_dim, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(con_dim), nn.ReLU(True),
                                nn.Conv2d(con_dim, con_dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(con_dim)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model2l = nn.Sequential(*model2l)
        self.model3l = nn.Sequential(*model3l)
        self.model_landmark_trans = nn.Sequential(*model_landmark_trans)

    def forward(self, input, land1, land2):
        """Standard forward"""
        x = self.model1(input)
        l1 = self.model_landmark_trans(land1)
        l2 = self.model_landmark_trans(land2)
        y = x.clone()

        for i in range(self.n_blocks):
            if (i + 1) % self.div == 0:
                x = self.model2[i](torch.cat([torch.cat([x, l1], dim=1), l2], dim=1))
            else:
                x = self.model2[i](x)
        out1 = self.model3(x)

        for i in range(self.n_blocks2):
            if (i + 1) % self.div == 0:
                y = self.model2l[i](torch.cat([torch.cat([y, l1], dim=1), l2], dim=1))
            else:
                y = self.model2l[i](y)
        out2 = self.model3l(y)

        return out1, out2

class Combiner(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=2, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Combiner, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetConditionTriGenerator2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        self.n_blocks = n_blocks
        self.div = 3
        super(ResnetConditionTriGenerator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        '''
        model11 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model11 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        '''

        mult = 2 ** n_downsampling
        model2 = []

        for i in range(n_blocks):       # add ResNet blocks
            if i == 0:
                model2 += [
                    ResnetBlock2(ngf * mult + ngf * mult, ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                use_dropout=use_dropout, use_bias=use_bias)]
            else:
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                       use_dropout=use_dropout, use_bias=use_bias)]

        model3 = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model3 += [nn.ReflectionPad2d(3)]
        model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model3 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        #self.model11 = nn.Sequential(*model11)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, input, warped):
        """Standard forward"""
        x = self.model1(input)
        #x1 = self.model11(warped)
        x1 = self.model1(warped)
        for i in range(self.n_blocks):
            if i == 0:
                x = self.model2[i](torch.cat([x, x1], dim=1))
            else:
                x = self.model2[i](x)
        out = self.model3(x)

        return out

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel=3):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, kernel)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel=3):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        pad = int((kernel-1)/2)
        if padding_type == 'reflect':#by default
            conv_block += [nn.ReflectionPad2d(pad)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(pad)]
        elif padding_type == 'zero':
            p = pad
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(pad)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(pad)]
        elif padding_type == 'zero':
            p = pad
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetBlock2(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias)
        self.shortcut = nn.Sequential(*[nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim_out)])

    def build_conv_block(self, dim_in, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.shortcut(x) + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Regressor(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, arch=1):
        super(Regressor, self).__init__()
        # if use BatchNorm2d, 
        # no need to use bias as BatchNorm2d has affine parameters

        self.arch = arch
        
        if arch == 1:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=2, padding=0, bias=use_bias),#11->5
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 1, kernel_size=5, stride=1, padding=0, bias=use_bias),#5->1
            ]
        elif arch == 2:
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=0, bias=use_bias),#11->9
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=0, bias=use_bias),#9->7
                norm_layer(ngf*2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=0, bias=use_bias),#7->5
                norm_layer(ngf*4),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*4, 1, kernel_size=5, stride=1, padding=0, bias=use_bias),#5->1
            ]
        elif arch == 3:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, 1, kernel_size=11, stride=1, padding=0, bias=use_bias),#11->1
            ]
        elif arch == 4:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*4, 1, kernel_size=11, stride=1, padding=0, bias=use_bias),#11->1
            ]
        elif arch == 5:
            use_bias = True
            sequence = [
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1, bias=use_bias),#11->11
                nn.LeakyReLU(0.2, True),
            ]
            fc = [
                nn.Linear(ngf*4*11*11, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1),
            ]
            self.fc = nn.Sequential(*fc)

        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        if self.arch <= 4:
            return self.model(x)
        else:
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
            
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class NLayerDiscriminatorCls(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, n_class=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminatorCls, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence1 = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence1 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        sequence2 = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence2 += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence2 += [
            nn.Conv2d(ndf * nf_mult, n_class, kernel_size=16, stride=1, padding=0, bias=use_bias)]


        self.model0 = nn.Sequential(*sequence)
        self.model1 = nn.Sequential(*sequence1)
        self.model2 = nn.Sequential(*sequence2)
        print(list(self.modules()))

    def forward(self, input):
        """Standard forward."""
        feat = self.model0(input)
        # patchGAN output (1 * 62 * 62)
        patch = self.model1(feat)
        # class output (3 * 1 * 1)
        classl = self.model2(feat)
        return patch, classl.view(classl.size(0), -1)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.moduleVggOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 2:3, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 0:1, :, :] * 255.0) - 122.67891434
        
        tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)
        
        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)
        
        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)
        
        tensorScoreOne = nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreTwo = nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreThr = nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFou = nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFiv = nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        
        return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))


class ULP(nn.Module):
    # upper landmark predictor
    def __init__(self, landmark_num):
        super(ULP, self).__init__()
        self.net = [nn.Linear(68 * 2, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, landmark_num * 2)]
        self.landmark_num = landmark_num

        self.net = nn.Sequential(*self.net)

    def forward(self, lm68):
        lm68_flatten = lm68.view(-1, 68 * 2)
        upper = self.net(lm68_flatten)
        out = upper.view(-1, self.landmark_num, 2)
        return out


class FaceLoss(nn.Module):
    def __init__(self, pretrained_path='./checkpoints/sphere20a_20171020.pth'):
        super(FaceLoss, self).__init__()
        if 'senet' in pretrained_path:
            self.net = senet50(include_top=False)
            self.load_senet_model(pretrained_path)
            self.height, self.width = 224, 224
        else:
            self.net = Sphere20a()
            self.load_sphere_model(pretrained_path)
            self.height, self.width = 112, 96

        self.net.eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        # from utils.demo_visualizer import MotionImitationVisualizer
        # self._visualizer = MotionImitationVisualizer('debug', ip='http://10.10.10.100', port=31102)

    def forward(self, imgs1, imgs2, kps1=None, kps2=None, bbox1=None, bbox2=None):
        """
        Args:
            imgs1:
            imgs2:
            kps1:
            kps2:
            bbox1:
            bbox2:

        Returns:

        """
        if kps1 is not None:
            head_imgs1 = self.crop_head_kps(imgs1, kps1)
        elif bbox1 is not None:
            head_imgs1 = self.crop_head_bbox(imgs1, bbox1)
        elif self.check_need_resize(imgs1):
            head_imgs1 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs1 = imgs1

        if kps2 is not None:
            head_imgs2 = self.crop_head_kps(imgs2, kps2)
        elif bbox2 is not None:
            head_imgs2 = self.crop_head_bbox(imgs2, bbox2)
        elif self.check_need_resize(imgs2):
            head_imgs2 = F.interpolate(imgs1, size=(self.height, self.width), mode='bilinear', align_corners=True)
        else:
            head_imgs2 = imgs2

        loss = self.compute_loss(head_imgs1, head_imgs2)

        # self._visualizer.vis_named_img('img2', imgs2)
        # self._visualizer.vis_named_img('head imgs2', head_imgs2)
        #
        # self._visualizer.vis_named_img('img1', imgs1)
        # self._visualizer.vis_named_img('head imgs1', head_imgs1)
        # import ipdb
        # ipdb.set_trace()

        #util.save_image(util.tensor2im(head_imgs1),'temp_head_imgs1.png')
        #util.save_image(util.tensor2im(head_imgs2),'temp_head_imgs2.png')

        return loss

    def compute_loss(self, img1, img2):
        """
        :param img1: (n, 3, 112, 96), [-1, 1]
        :param img2: (n, 3, 112, 96), [-1, 1], if it is used in training,
                     img2 is reference image (GT), use detach() to stop backpropagation.
        :return:
        """
        f1, f2 = self.net(img1), self.net(img2)

        loss = 0.0
        for i in range(len(f1)):
            loss += self.criterion(f1[i], f2[i].detach())

        return loss

    def check_need_resize(self, img):
        return img.shape[2] != self.height or img.shape[3] != self.width

    def crop_head_bbox(self, imgs, bboxs):
        """
        Args:
            bboxs: (N, 4), 4 = [lt_x, lt_y, rt_x, rt_y]

        Returns:
            resize_image:
        """
        bs, _, ori_h, ori_w = imgs.shape

        head_imgs = []

        for i in range(bs):
            x1, x2, y1, y2 = bboxs[i]
            box_size = x2-x1
            head = torch.ones((1,3,box_size,box_size)).to(imgs.device)
            head[:,:,max(0,y1)-y1:min(y2,ori_h)-y1,max(0,x1)-x1:min(ori_w,x2)-x1] = imgs[i:i+1,:,max(0,y1):min(y2,ori_h),max(0,x1):min(ori_w,x2)]
            #head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    def crop_head_kps(self, imgs, kps):
        """
        :param imgs: (N, C, H, W)
        :param kps: (N, 19, 2)
        :return:
        """
        bs, _, ori_h, ori_w = imgs.shape

        rects = self.find_head_rect(kps, ori_h, ori_w)
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i]
            head = imgs[i:i+1, :, min_y:max_y, min_x:max_x]  # (1, c, h', w')
            head = F.interpolate(head, size=(self.height, self.width), mode='bilinear', align_corners=True)
            head_imgs.append(head)

        head_imgs = torch.cat(head_imgs, dim=0)

        return head_imgs

    @staticmethod
    @torch.no_grad()
    def find_head_rect(kps, height, width):
        NECK_IDS = 12

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = torch.zeros_like(necks)
        ones = torch.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = torch.min(kps[:, NECK_IDS:, 0] - 0.05, dim=1)
        min_x = torch.max(min_x, zeros)

        max_x, _ = torch.max(kps[:, NECK_IDS:, 0] + 0.05, dim=1)
        max_x = torch.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = torch.min(kps[:, NECK_IDS:, 1] - 0.05, dim=1)
        min_y = torch.max(min_y, zeros)

        max_y, _ = torch.max(kps[:, NECK_IDS:, 1], dim=1)
        max_y = torch.min(max_y, ones)

        min_x = (min_x * width).long()      # (T, 1)
        max_x = (max_x * width).long()      # (T, 1)
        min_y = (min_y * height).long()     # (T, 1)
        max_y = (max_y * height).long()     # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = torch.stack((min_x, max_x, min_y, max_y), dim=1)

        # import ipdb
        # ipdb.set_trace()

        return rects

    def load_senet_model(self, pretrain_model):
        # saved_data = torch.load(pretrain_model, encoding='latin1')
        with open(pretrain_model, 'rb') as f:
            saved_data = pickle.load(f, encoding='latin1')
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc'):
                continue
            save_weights_dict[key] = torch.from_numpy(val)

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))

    def load_sphere_model(self, pretrain_model):
        saved_data = torch.load(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith('fc6'):
                continue
            save_weights_dict[key] = val

        self.net.load_state_dict(save_weights_dict)

        print('load face model from {}'.format(pretrain_model))