import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
from models.mobilefacenet import MobileFaceNet
import models.sparse_image_warp as sparse_image_warp
import cv2
import numpy as np
import math

import json
import argparse
import intrinsic_flow_models.networks as ifm_networks
from .modnet import MODNet
from torch.nn import functional as F

def kp_to_map(img_sz, kps, mode='binary', radius=4):
    '''
    Keypoint cordinates to heatmap map.
    Input:
        img_size (w,h): size of heatmap
        kps (N,2): (x,y) cordinates of N keypoints
        mode: 'gaussian' or 'binary'
        radius: radius of each keypoints in heatmap
    Output:
        m (h,w,N): encoded heatmap
    '''
    w, h = img_sz
    x_grid, y_grid = np.meshgrid(range(w), range(h), indexing = 'xy')
    m = []
    for x, y in kps:
        if x == -1 or y == -1:
            m.append(np.zeros((h, w)).astype(np.float32))
        else:
            if mode == 'gaussian':
                m.append(np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(radius**2)).astype(np.float32))
            elif mode == 'binary':
                m.append(((x_grid-x)**2 + (y_grid-y)**2 <= radius**2).astype(np.float32))
            else:
                raise NotImplementedError()
    m = np.stack(m, axis=2)
    return torch.from_numpy(m.transpose((2,0,1)))
    
def kp_to_map_some(img_sz, kps, mode='binary', radius=4):
    res = []
    for i in kps:
        resi = kp_to_map(img_sz, i, mode='binary', radius=4)
        res.append(resi)
    return torch.stack(res, 0)
    
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def load_flow_network(model_id='FlowReg_id_flow_faces', epoch='best', gpu_ids=[0]):
    from intrinsic_flow_models.flow_regression_model import FlowRegressionModel
    opt_dict = load_json(os.path.join('checkpoints', model_id, 'train_opt.json'))
    opt = argparse.Namespace(**opt_dict)
    opt.gpu_ids = gpu_ids
    opt.is_train = False # prevent loading discriminator, optimizer...
    opt.which_epoch = epoch
    # create network
    model = FlowRegressionModel()
    model.initialize(opt)
    return model.netF
    
def flow_network_warp(netF, real_A, lm1, lm2):
    with torch.no_grad():
        real_A224 = F.interpolate(real_A, size = (224,224), mode='bilinear', align_corners=True)
        j1 = kp_to_map_some((224,224), lm1.cpu().numpy()*7/8)
        j2 = kp_to_map_some((224,224), lm2.cpu().numpy()*7/8)
        input_F = torch.cat([j1,j2],1).cuda()
        flow_out, vis_out, _, _ = netF(input_F)
        output = {}
        output['vis_out'] = vis_out.argmax(dim=1, keepdim=True).float()
        output['mask_out'] = (output['vis_out']<2).float()
        output['flow_out'] = flow_out * 20. * output['mask_out']
        #output['img_warp'] = ifm_networks.warp_acc_flow(real_A224, output['flow_out'], mask=output['mask_out'])
    
        warp_flow = F.interpolate(output['flow_out']/7*8, size = (256,256), mode='bilinear', align_corners=True)
        res_mask = F.interpolate(output['mask_out'], size = (256,256), mode='bilinear', align_corners=True)
    return warp_flow, res_mask
    
def flow_to_grid(x, flow):
    bsz, c, h, w = x.size()
    # mesh grid
    xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
    yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
    xx = xx.view(1,1,h,w).repeat(bsz,1,1,1)
    yy = yy.view(1,1,h,w).repeat(bsz,1,1,1)
    grid = torch.cat((xx,yy), dim=1).float()
    grid = grid + flow
    # scale to [-1, 1]
    grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0

    grid = grid.permute(0,2,3,1)
    return grid

from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
def get_grid_boundary_convexhull_demask(xx_grid, ux_alpha):
    img_size = xx_grid.shape[1]
    ux_alpha_demask = ux_alpha.unsqueeze(0)
    increase_grid_x = torch.arange(0,1,step=1./img_size).cuda().view(1,img_size,1)
    x_max = (increase_grid_x + ux_alpha_demask[0]).argmax(1)[0].cpu().numpy()
    x_min = (increase_grid_x - ux_alpha_demask[0]).argmin(1)[0].cpu().numpy()
    rank = []
    value = []
    for i in range(img_size):
        if x_max[i] != 255 and x_min[i] != 0:
            rank.append([x_min[i], i])
            rank.append([x_max[i], i])
    rank = np.array(rank)
    vh = ConvexHull(rank).vertices.tolist()
    rank = rank[vh,:]
    for i in rank:
        value.append(xx_grid[:, i[0], i[1]].cpu().numpy())
        delta = xx_grid[:, i[0], i[1]].cpu().numpy() - 2*i[[1,0]]/255+1
    value = np.array(value)
    return rank, value, ux_alpha_demask

def param_make_grid(left_up = [2.5,2.5], length = 250, step = 5):
    steps = np.linspace(0, length+step, step)
    edges_fix = np.array([[0, 0], [255, 255], [0, 255], [255, 0]])
    edges = [[0, 0], [length, length], [0, length], [length, 0]]
    for i in steps:
        edges.append([0, i])
        edges.append([length, i])
        edges.append([i, 0])
        edges.append([i, length])
    edges = np.array(edges)
    edges[:,0] = edges[:,0] + left_up[0]
    edges[:,1] = edges[:,1] + left_up[1]
    res = np.concatenate([edges_fix, edges], 0)
    return res

def cal_warp(rank, value):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    edges = param_make_grid()
    edges_value = 2*edges/255-1
    edges_value = edges_value[:,[1,0]]
    destination = np.concatenate((rank, edges))
    source = np.concatenate((value, edges_value))
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    return grid_z

def iw_convexhull_grid(real_A, iw_flow, real_A_if_mask, device):
    # assume batch_size is 1
    iw_flow_grid = flow_to_grid(real_A, iw_flow) # [1, 256, 256, 2]
    rank, value, mask = get_grid_boundary_convexhull_demask(iw_flow_grid.permute(0, 3, 1, 2)[0], real_A_if_mask[0])
    gridz = cal_warp(rank, value)
    gridz = torch.from_numpy(gridz).unsqueeze(0).float() # [1, 256, 256, 2]
    mask1 = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 2) # [1, 1, 256, 256] -> [1, 256, 256, 2]
    iw_flow_convexhull_grid = gridz.to(device) * (1 - mask1) + iw_flow_grid * mask1
    return iw_flow_grid, iw_flow_convexhull_grid

class GeomGMIFWForeModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(dataset_mode='umlvd_ifw')
        parser.set_defaults(netG='resnet_9blocks_rcatland3')
        parser.add_argument('--netg_resb_div', type=int, default=3, help='div')
        parser.add_argument('--netg_resb_disp', type=int, default=1, help='disp')
        if is_train:
            parser.add_argument('--lambda_geom', type=float, default=5.0, help='weight for geom loss (landmark)')
            parser.add_argument('--lambda_geom_lipline', type=float, default=0.0, help='weight for geom loss for lipline')

            parser.add_argument('--max_offset', type=float, default=3, help='max offset')
            
            parser.add_argument('--lambda_G_A_l', type=float, default=0.5, help='weight for local GAN loss in G')
        # for masks
        parser.add_argument('--use_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--use_eye_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--use_lip_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--mask_type', type=int, default=3, help='use mask type, 0 outside black, 1 outside white')
        parser.add_argument('--blendbg', type=int, default=0, help='blend with bg line drawing')
        if is_train:
            # add constraint
            # 1. identity loss
            parser.add_argument('--identity_loss', type=int, default=2, help='use identity_loss, =1 calc iden feature for realA fakeB, =2 calc iden feature for fakeB_static fakeB')
            parser.add_argument('--face_recog_model', type=str, default='./checkpoints/sphere20a_20171020.pth', help='path for face recognition model')
            parser.add_argument('--lambda_face', type=float, default=5.0, help='weight for identity loss')
            # 2. warp loss
            parser.add_argument('--warp_loss', type=int, default=2, help='use warp as approximate GT, =1 calc warp in dataset, =2 calc warp in forward')
            parser.add_argument('--lambda_warp', type=float, default=5.0, help='weight for warp loss')
            # 3. coherent loss based on warping
            parser.add_argument('--lambda_warp_inter', type=float, default=5.0, help='weight for inter-frame warp loss')
            # 4. coherent loss based on coherent discriminator
            parser.add_argument('--coherent', type=int, default=1, help='use coherent discriminator')
            parser.add_argument('--lambda_G_A_coh', type=float, default=0.5, help='weight for coherent GAN loss in G')
            parser.add_argument('--coh_use_more', type=int, default=2, help='use more type of samples for coherent discriminator')
            parser.add_argument('--check_fakeb2_in_backwardD', type=int, default=1, help='check_fakeb2_in_backwardD')
            # 6. how to set target landmark
            parser.add_argument('--select_target12_thre', type=float, default=0.2, help='the probability of using drawing films for targetB1,B2_lm')
            parser.add_argument('--select_noniden_thre', type=float, default=0.9, help='the probability of using non-iden landmark as target')
            # 7. more weighr for lip
            parser.add_argument('--more_weight_for_lip', type=int, default=0, help='use more weight for lip')
            # 9. 2/3 face
            parser.add_argument('--rx', type=float, default=0.15, help='rx for win')
            parser.add_argument('--ry', type=float, default=0.2, help='ry for win')
            parser.add_argument('--rs', type=float, default=0.7, help='ry for win')



        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'geom_B']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'realA_static_warp', 'realA_static_warp2', 'real_A_lm', 'target_B_lm', 'fake_B', 'target_B2_lm', 'fake_B2']
        visual_names_B = ['real_B', 'real_B_lm']
        if self.isTrain:
            visual_names_A = ['real_A', 'real_A_fore', 'realA_static_warp', 'realA_static_warp2', 'real_A_if_face', 'real_A_if_mask',  'real_A_lm', 'target_B_lm', 'fake_B', 'fake_B_lm_68_vis', 'fake_B_lm_68_vist', 'target_B2_lm', 'fake_B2', 'fake_B2_lm_68_vis', 'fake_B2_lm_68_vist']
            visual_names_B = ['real_B', 'real_B_lm']
        if self.opt.blendbg:
            visual_names_A.append('fake_B_fore')
            visual_names_A.append('fake_B2_fore')
            visual_names_A.append('mask')
            visual_names_A.append('mask1')
            visual_names_A.append('mask2')
            if self.opt.blendbg == 2:
                visual_names_A.append('fakeB_static_bg')
        if self.isTrain and self.opt.lambda_geom_lipline > 0:
            visual_names_A.append('liplinemask1')
            visual_names_A.append('liplinemask2')
            self.loss_names += ['geom_B_lipline']
        if self.isTrain and self.opt.warp_loss:
            if self.opt.warp_loss == 2:
                visual_names_A.append('fakeB_static')
            visual_names_A.append('fakeB_static_warp')
            self.loss_names += ['warp_B']
        if self.isTrain:
            visual_names_A.append('fake_B_warp')
            self.loss_names += ['warp_inter1']
        if self.isTrain and self.opt.identity_loss:
            self.loss_names += ['iden_B']
        if self.isTrain and self.opt.coherent:
            visual_names_A.append('real_B1_lm_68_vist')
            visual_names_A.append('real_B2_lm_68_vist')
            if self.opt.coh_use_more:
                visual_names_A.append('real_B3')
                visual_names_A.append('real_B4')
            self.loss_names += ['D_A_coh', 'G_A_coh']
        if self.isTrain and self.opt.use_mask:
            visual_names_A.append('fake_B_l')
            visual_names_A.append('fake_B2_l')
            visual_names_A.append('real_B_l')
            self.loss_names += ['D_A_l', 'G_A_l']
        if self.isTrain and self.opt.use_eye_mask:
            visual_names_A.append('fake_B_le')
            visual_names_A.append('fake_B2_le')
            visual_names_A.append('real_B_le')
            self.loss_names += ['D_A_le', 'G_A_le']
        if self.isTrain and self.opt.use_lip_mask:
            visual_names_A.append('fake_B_ll')
            visual_names_A.append('fake_B2_ll')
            visual_names_A.append('real_B_ll')
            self.loss_names += ['D_A_ll', 'G_A_ll']
        if self.isTrain:
            self.loss_names += ['G']
        
        if not self.isTrain and self.opt.use_mask:
            visual_names_A.append('fake_B_l')
            visual_names_A.append('real_B_l')
        if not self.isTrain and self.opt.use_eye_mask:
            visual_names_A.append('fake_B_le')
            visual_names_A.append('real_B_le')
        if not self.isTrain and self.opt.use_lip_mask:
            visual_names_A.append('fake_B_ll')
            visual_names_A.append('real_B_ll')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        print(self.visual_names)
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
            if self.opt.use_mask:
                self.model_names += ['D_A_l']
            if self.opt.use_eye_mask:
                self.model_names += ['D_A_le']
            if self.opt.use_lip_mask:
                self.model_names += ['D_A_ll']
            if self.opt.coherent:
                self.model_names += ['D_A_coh']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        self.gpu_p = self.opt.gpu_ids_p[0]
        self.gpu = self.opt.gpu_ids[0]
        
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        div=self.opt.netg_resb_div, disp=self.opt.netg_resb_disp)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.opt.gpu_ids_p)
            if self.opt.use_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_l = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.opt.gpu_ids_p)
            if self.opt.use_eye_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_le = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.opt.gpu_ids_p)
            if self.opt.use_lip_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_ll = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.opt.gpu_ids_p)
            if self.opt.coherent:
                self.netD_A_coh = networks.define_D(opt.output_nc*2, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.opt.gpu_ids_p)
        

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda(self.gpu_p)  # define GAN loss.
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.opt.use_mask:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif not self.opt.use_eye_mask:
                D_params = list(self.netD_A.parameters()) + list(self.netD_A_l.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif not self.opt.use_lip_mask:
                D_params = list(self.netD_A.parameters()) + list(self.netD_A_l.parameters()) + list(self.netD_A_le.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif not self.opt.coherent:
                D_params = list(self.netD_A.parameters()) + list(self.netD_A_l.parameters()) + list(self.netD_A_le.parameters()) + list(self.netD_A_ll.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                D_params = list(self.netD_A.parameters()) + list(self.netD_A_l.parameters()) + list(self.netD_A_le.parameters()) + list(self.netD_A_ll.parameters()) + list(self.netD_A_coh.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)    

            self.mobilefacenet = MobileFaceNet([112, 112],136)   
            checkpoint = torch.load('checkpoints/mobilefacenet_model_best.pth.tar', map_location=str(self.device))
            self.mobilefacenet.load_state_dict(checkpoint['state_dict'])
            self.mobilefacenet.eval()
            self.mobilefacenet.cuda(self.gpu_p)

            self.modnet = MODNet(backbone_pretrained=False)
            self.modnet = torch.nn.DataParallel(self.modnet).cuda(self.gpu)
            self.modnet.load_state_dict(torch.load('checkpoints/modnet_photographic_portrait_matting.ckpt'))
            self.modnet.eval()

            if self.opt.identity_loss:
                self.faceidenloss = networks.FaceLoss(pretrained_path=self.opt.face_recog_model)
                self.faceidenloss.cuda(self.gpu_p)

        cs = self.opt.crop_size
        csh = cs//2
        self.edges = torch.Tensor([[[0, 0], [cs-1, cs-1], [0, cs-1], [cs-1, 0], [0, csh-1], [csh-1, 0], [csh-1, cs-1], [cs-1, csh-1]]]).to(self.device)
        self.radius = 5 if cs == 512 else 3
        self.faceLmarkLookup = np.load('./faceLmarkLookup.npy')
        self.thickness = 4 if cs == 512 else 2
        
        self.netF = load_flow_network()
        self.netF.eval()
        self.netF.cuda()
        
    def get_lm(self, input, win, out_size=112):
        # assume batch_size is 1
        # crop win, resize to 112
        bs,c,h,w = input.shape
        height=width=self.opt.crop_size
        x1 = win[0,0]
        x2 = win[0,1]
        y1 = win[0,2]
        y2 = win[0,3]
        box_size = x2-x1
        box = torch.ones((bs,c,box_size,box_size))
        box[:,:,max(0,y1)-y1:min(y2,height)-y1,max(0,x1)-x1:min(width,x2)-x1] = input[:,:,max(0,y1):min(y2,height),max(0,x1):min(width,x2)]
        #input = input[:,:,win[0,2]:win[0,3],win[0,0]:win[0,1]]
        if box.shape[1] == 3:
          box = box[:,[2,1,0],:,:]
        else:
          box = box.repeat(1,3,1,1)
        box = torch.nn.functional.interpolate(input=box, size=(out_size, out_size), mode='bicubic', align_corners=False)
        # extract landmark
        bs = box.shape[0]
        landmark = self.mobilefacenet((box.cuda(self.gpu_p)+1)*0.5)[0] # scale to [0,1]
        landmark = landmark.view((bs,68,2))
        # reproject
        landmark[:,:,0] = landmark[:,:,0] * (win[0,1]-win[0,0]) + win[0,0]
        landmark[:,:,1] = landmark[:,:,1] * (win[0,3]-win[0,2]) + win[0,2]
        return landmark.cuda(self.gpu)
    
    def get_lmvis(self, tensor_im, lm, win, hradius=3):
        # assume batch_size is 1
        tensor_vis = tensor_im.detach().clone()
        if tensor_vis.shape[1] == 1:
            tensor_vis = tensor_vis.repeat(1,3,1,1)
        lmvis = lm.detach().clone().cpu().data.numpy()
        for k in range(lm.shape[1]):
            x = int(round(lmvis[0,k,0]))
            y = int(round(lmvis[0,k,1]))
            tensor_vis[:,0,y-hradius:y+hradius,x-hradius:x+hradius] = 1
            tensor_vis[:,1:,y-hradius:y+hradius,x-hradius:x+hradius] = -1
        tensor_vis[:,0,win[0,2]-hradius:win[0,2]+hradius,win[0,0]-hradius:win[0,1]+hradius] = 1
        tensor_vis[:,1:,win[0,2]-hradius:win[0,2]+hradius,win[0,0]-hradius:win[0,1]+hradius] = -1
        tensor_vis[:,0,win[0,3]-hradius:win[0,3]+hradius,win[0,0]-hradius:win[0,1]+hradius] = 1
        tensor_vis[:,1:,win[0,3]-hradius:win[0,3]+hradius,win[0,0]-hradius:win[0,1]+hradius] = -1
        tensor_vis[:,0,win[0,2]-hradius:win[0,3]+hradius,win[0,0]-hradius:win[0,0]+hradius] = 1
        tensor_vis[:,1:,win[0,2]-hradius:win[0,3]+hradius,win[0,0]-hradius:win[0,0]+hradius] = -1
        tensor_vis[:,0,win[0,2]-hradius:win[0,3]+hradius,win[0,1]-hradius:win[0,1]+hradius] = 1
        tensor_vis[:,1:,win[0,2]-hradius:win[0,3]+hradius,win[0,1]-hradius:win[0,1]+hradius] = -1
        return tensor_vis
        
    def cat(self, A, lA, lB):
        return torch.cat((torch.cat((A,lA), 1),lB), 1)
        


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.realA_static_warp = input['realA_static_warp'].to(self.device)
        self.realA_static_warp2 = input['realA_static_warp2'].to(self.device)
        self.warp_motion = input['warp_motion']
        self.warp_motion2 = input['warp_motion2']
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_lm = input['A_lm'].to(self.device)
        self.real_B_lm = input['B_lm'].to(self.device)
        self.real_A_lm_68 = input['A_lm_68'].to(self.device)
        self.real_B_lm_68 = input['B_lm_68'].to(self.device)
        self.target_B_lm = input['tB_lm'].to(self.device)
        self.target_B2_lm = input['tB2_lm'].to(self.device)
        self.target_B_lm_68 = input['tB_lm_68'].to(self.device)
        self.target_B2_lm_68 = input['tB2_lm_68'].to(self.device)
        self.winA = input['winA']
        # winB -- tB_lm, winB2 -- tB2_lm, winBr -- B_lm
        self.winB = input['winB']
        self.winB2 = input['winB2']
        self.winBr = input['winBr']
        self.image_paths = input['image_paths']
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']
        if self.isTrain and self.opt.warp_loss == 1:
            self.fakeB_static_warp = input['fakeB_static_warp'].to(self.device)
        elif self.isTrain and (self.opt.warp_loss == 2 or self.opt.identity_loss == 2):
            self.fakeB_static = input['fakeB_static'].to(self.device)
        if self.opt.use_mask:
            self.Br_mask = input['Br_mask'].to(self.device)
            self.B_mask = input['B_mask'].to(self.device)
            self.B2_mask = input['B2_mask'].to(self.device)
        if self.opt.use_eye_mask:
            self.Br_maske = input['Br_maske'].to(self.device)
            self.B_maske = input['B_maske'].to(self.device)
            self.B2_maske = input['B2_maske'].to(self.device)
        if self.opt.use_lip_mask:
            self.Br_maskl = input['Br_maskl'].to(self.device)
            self.B_maskl = input['B_maskl'].to(self.device)
            self.B2_maskl = input['B2_maskl'].to(self.device)
        if self.opt.coherent:
            # load drawings of two consecutive frames, only used for coherent discriminator
            self.real_B1 = input['B1'].to(self.device)
            self.real_B2 = input['B2'].to(self.device)
            self.real_B1_lm_68 = input['B1_lm_68'].to(self.device)
            self.real_B2_lm_68 = input['B2_lm_68'].to(self.device)
            self.winBr1 = input['winBr1'].to(self.device)
            self.winBr2 = input['winBr2'].to(self.device)
            self.B1_paths = input['B1_path']
            if self.opt.coh_use_more:
                self.real_B3 = input['B3'].to(self.device)
                self.real_B4 = input['B4'].to(self.device)
        #intrinsic flow
        self.iw_flow, self.real_A_if_mask = flow_network_warp(self.netF, self.real_A, self.real_A_lm_68, self.target_B_lm_68)
        self.real_A_if_face = ifm_networks.warp_acc_flow(self.real_A, self.iw_flow, mask = self.real_A_if_mask)
        self.iw_flow2, self.real_A_if_mask2 = flow_network_warp(self.netF, self.real_A, self.real_A_lm_68, self.target_B2_lm_68)

    def getlipline(self,lands):
        liplinemask = np.zeros((self.opt.crop_size,self.opt.crop_size))
        lands = lands[0,:,:].cpu().numpy()
        for refpts in self.faceLmarkLookup[:20]:
            start_point = (lands[refpts[0],0],lands[refpts[0],1])
            end_point = (lands[refpts[1],0],lands[refpts[1],1])
            frame = cv2.line(liplinemask, start_point, end_point, 255, self.thickness)
        liplinemask = torch.from_numpy(liplinemask)[None,None,...].float().to(self.device) / 255.
        return liplinemask

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        _, _, matte = self.modnet(self.real_A, True)
        mask = (matte > 0.5).float()
        self.mask = mask
        if not self.opt.blendbg:
            self.real_A = ((self.real_A/2+0.5)*mask + 1-mask)*2-1
            self.real_A_fore = self.real_A
            self.fakeB_static = ((self.fakeB_static/2+0.5)*mask + 1-mask)*2-1
        else:
            self.real_A_fore = ((self.real_A/2+0.5)*mask + 1-mask)*2-1

        # change both style and geom
        self.fake_B = self.netG_A(self.real_A_fore, self.real_A_lm, self.target_B_lm, self.warp_motion, self.iw_flow, self.real_A_if_mask)
        self.fake_B2 = self.netG_A(self.real_A_fore, self.real_A_lm, self.target_B2_lm, self.warp_motion2, self.iw_flow2, self.real_A_if_mask2)
        
        if self.opt.blendbg:
            self.real_A_lm_681 = self.real_A_lm_68
            self.target_B_lm_681 = self.target_B_lm_68
            self.target_B2_lm_681 = self.target_B2_lm_68
            self.mask1 = sparse_image_warp.sparse_image_warp(mask.permute(0,2,3,1), self.real_A_lm_681[:,:,[1,0]], self.target_B_lm_681[:,:,[1,0]])[0].permute(0,3,1,2)
            self.mask2 = sparse_image_warp.sparse_image_warp(mask.permute(0,2,3,1), self.real_A_lm_681[:,:,[1,0]], self.target_B2_lm_681[:,:,[1,0]])[0].permute(0,3,1,2)
            
            self.fake_B_fore = self.fake_B.clone()
            self.fake_B = ((self.fake_B/2+0.5)*self.mask1 + (self.fakeB_static/2+0.5)*(1-self.mask1))*2-1
            self.fake_B2_fore = self.fake_B2.clone()
            self.fake_B2 = ((self.fake_B2/2+0.5)*self.mask2 + (self.fakeB_static/2+0.5)*(1-self.mask2))*2-1

        # fakeB should have same geometry as target_B_landmark 
        if self.opt.use_mask:
            self.fake_B_l = self.masked(self.fake_B,self.B_mask)
            self.fake_B2_l = self.masked(self.fake_B2,self.B2_mask)
            self.real_B_l = self.masked(self.real_B,self.Br_mask)
        if self.opt.use_eye_mask:
            self.fake_B_le = self.masked(self.fake_B,self.B_maske)
            self.fake_B2_le = self.masked(self.fake_B2,self.B2_maske)
            self.real_B_le = self.masked(self.real_B,self.Br_maske)
        if self.opt.use_lip_mask:
            self.fake_B_ll = self.masked(self.fake_B,self.B_maskl)
            self.fake_B2_ll = self.masked(self.fake_B2,self.B2_maskl)
            self.real_B_ll = self.masked(self.real_B,self.Br_maskl)
        if self.isTrain and self.opt.warp_loss == 2:
            if not hasattr(self, "real_A_lm_681"):
                # similar to cal_warp_path, add edges
                self.real_A_lm_681 = torch.cat((self.real_A_lm_68, self.edges), 1)
            if not hasattr(self, "target_B_lm_681"):
                self.target_B_lm_681 = torch.cat((self.target_B_lm_68, self.edges), 1)
            # warp
            self.fakeB_static_warp = sparse_image_warp.sparse_image_warp(self.fakeB_static.permute(0,2,3,1), self.real_A_lm_681[:,:,[1,0]], self.target_B_lm_681[:,:,[1,0]])[0].permute(0,3,1,2)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_basic2(self, netD, real, fake1, fake2):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake1 = netD(fake1.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake1, False)
        pred_fake2 = netD(fake2.detach())
        loss_D_fake2 = self.criterionGAN(pred_fake2, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake1 + loss_D_fake2) / 3.0
        loss_D.backward()
        return loss_D

    def backward_D_basic3(self, netD, real, fake1, fake2):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake1 = netD(fake1.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake1, False)
        pred_fake2 = netD(fake2.detach())
        loss_D_fake2 = self.criterionGAN(pred_fake2, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + (loss_D_fake1 + loss_D_fake2) / 2.0) / 2.0
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        if not self.opt.check_fakeb2_in_backwardD:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B.cuda(self.gpu_p), self.fake_B.cuda(self.gpu_p))
        else:
            self.loss_D_A = self.backward_D_basic3(self.netD_A, self.real_B.cuda(self.gpu_p), self.fake_B.cuda(self.gpu_p), self.fake_B2.cuda(self.gpu_p))
    
    def backward_D_A_l(self):
        """Calculate GAN loss for discriminator D_A_l"""
        if not self.opt.check_fakeb2_in_backwardD:
            self.loss_D_A_l = self.backward_D_basic(self.netD_A_l, self.real_B_l.cuda(self.gpu_p), self.fake_B_l.cuda(self.gpu_p))
        else:
            self.loss_D_A_l = self.backward_D_basic3(self.netD_A_l, self.real_B_l.cuda(self.gpu_p), self.fake_B_l.cuda(self.gpu_p), self.fake_B2_l.cuda(self.gpu_p))

    def backward_D_A_le(self):
        """Calculate GAN loss for discriminator D_A_le"""
        if not self.opt.check_fakeb2_in_backwardD:
            self.loss_D_A_le = self.backward_D_basic(self.netD_A_le, self.real_B_le.cuda(self.gpu_p), self.fake_B_le.cuda(self.gpu_p))
        else:
            self.loss_D_A_le = self.backward_D_basic3(self.netD_A_le, self.real_B_le.cuda(self.gpu_p), self.fake_B_le.cuda(self.gpu_p), self.fake_B2_le.cuda(self.gpu_p))
    
    def backward_D_A_ll(self):
        """Calculate GAN loss for discriminator D_A_ll"""
        if not self.opt.check_fakeb2_in_backwardD:
            self.loss_D_A_ll = self.backward_D_basic(self.netD_A_ll, self.real_B_ll.cuda(self.gpu_p), self.fake_B_ll.cuda(self.gpu_p))
        else:
            self.loss_D_A_ll = self.backward_D_basic3(self.netD_A_ll, self.real_B_ll.cuda(self.gpu_p), self.fake_B_ll.cuda(self.gpu_p), self.fake_B2_ll.cuda(self.gpu_p))
    
    def backward_D_A_coh(self):
        """Calculate GAN loss for discriminator D_A_ll"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B2 = self.fake_B_pool.query(self.fake_B2)
        if not self.opt.coh_use_more:
            self.loss_D_A_coh = self.backward_D_basic(self.netD_A_coh, torch.cat((self.real_B1,self.real_B2),1).cuda(self.gpu_p), torch.cat((fake_B,fake_B2),1).cuda(self.gpu_p))
        else:
            self.loss_D_A_coh = self.backward_D_basic2(self.netD_A_coh, torch.cat((self.real_B1,self.real_B2),1).cuda(self.gpu_p), torch.cat((fake_B,fake_B2),1).cuda(self.gpu_p), torch.cat((self.real_B3,self.real_B4),1).cuda(self.gpu_p))
    
    def update_process(self, epoch):
        self.process = (epoch - 1) / float(self.opt.niter_decay + self.opt.niter)

    def backward_G(self):
        """Calculate the loss for generators G_A"""
        lambda_G_A_l = self.opt.lambda_G_A_l
        lambda_G_A_coh = self.opt.lambda_G_A_coh
        lambda_geom = self.opt.lambda_geom
        lambda_warp = self.opt.lambda_warp
        lambda_warp_inter = self.opt.lambda_warp_inter
        

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B.cuda(self.gpu_p)), True).cuda(self.gpu)
        self.loss_G_A = self.loss_G_A + self.criterionGAN(self.netD_A(self.fake_B2.cuda(self.gpu_p)), True).cuda(self.gpu)
        if self.opt.use_mask:
            self.loss_G_A_l = self.criterionGAN(self.netD_A_l(self.fake_B_l.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
            self.loss_G_A_l = self.loss_G_A_l + self.criterionGAN(self.netD_A_l(self.fake_B2_l.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
        if self.opt.use_eye_mask:
            self.loss_G_A_le = self.criterionGAN(self.netD_A_le(self.fake_B_le.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
            self.loss_G_A_le = self.loss_G_A_le + self.criterionGAN(self.netD_A_le(self.fake_B2_le.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
        if self.opt.use_lip_mask:
            self.loss_G_A_ll = self.criterionGAN(self.netD_A_ll(self.fake_B_ll.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
            self.loss_G_A_ll = self.loss_G_A_ll + self.criterionGAN(self.netD_A_ll(self.fake_B2_ll.cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_l
        if self.opt.coherent:
            self.loss_G_A_coh = self.criterionGAN(self.netD_A_coh(torch.cat((self.fake_B,self.fake_B2),1).cuda(self.gpu_p)), True).cuda(self.gpu) * lambda_G_A_coh
        

        # Geom loss || target_B_lm - LM(fake_B) ||
        csize = self.opt.crop_size
        self.fake_B_lm_68 = self.get_lm(self.fake_B, self.winB)
        self.fake_B2_lm_68 = self.get_lm(self.fake_B2, self.winB2)
        if self.opt.more_weight_for_lip != 2:
            self.loss_geom_B = torch.nn.MSELoss()(self.fake_B_lm_68/csize,self.target_B_lm_68[:,:68,:]/csize) * lambda_geom
            self.loss_geom_B = self.loss_geom_B + torch.nn.MSELoss()(self.fake_B2_lm_68/csize,self.target_B2_lm_68[:,:68,:]/csize) * lambda_geom
        else:
            self.loss_geom_B = torch.nn.MSELoss()(self.fake_B_lm_68[:,:48,:]/csize,self.target_B_lm_68[:,:48,:]/csize) * lambda_geom
            self.loss_geom_B = self.loss_geom_B + torch.nn.MSELoss()(self.fake_B_lm_68[:,48:68,:]/csize,self.target_B_lm_68[:,48:68,:]/csize) * lambda_geom * 2
            self.loss_geom_B = self.loss_geom_B + torch.nn.MSELoss()(self.fake_B2_lm_68[:,:48,:]/csize,self.target_B2_lm_68[:,:48,:]/csize) * lambda_geom
            self.loss_geom_B = self.loss_geom_B + torch.nn.MSELoss()(self.fake_B2_lm_68[:,48:68,:]/csize,self.target_B2_lm_68[:,48:68,:]/csize) * lambda_geom * 2
        
        if self.opt.lambda_geom_lipline > 0:
            self.liplinemask1 = self.getlipline(self.target_B_lm_68)
            self.liplinemask2 = self.getlipline(self.target_B2_lm_68)
            self.loss_geom_B_lipline = torch.mean((self.fake_B - (-1)) * self.liplinemask1) * self.opt.lambda_geom_lipline
            self.loss_geom_B_lipline = self.loss_geom_B_lipline + torch.mean((self.fake_B2 - (-1)) * self.liplinemask2) * self.opt.lambda_geom_lipline
        
        
        # validate lm extract and trans
        self.fake_B_lm_68_vis = self.get_lmvis(self.fake_B, self.fake_B_lm_68, self.winB)
        self.fake_B_lm_68_vist = self.get_lmvis(self.fake_B, self.target_B_lm_68, self.winB)

        self.fake_B2_lm_68_vis = self.get_lmvis(self.fake_B2, self.fake_B2_lm_68, self.winB2)
        self.fake_B2_lm_68_vist = self.get_lmvis(self.fake_B2, self.target_B2_lm_68, self.winB2)
        
        if self.opt.coherent:
            self.real_B1_lm_68_vist = self.get_lmvis(self.real_B1, self.real_B1_lm_68, self.winBr1)
            self.real_B2_lm_68_vist = self.get_lmvis(self.real_B2, self.real_B2_lm_68, self.winBr2)

        
        if self.opt.warp_loss:
            self.loss_warp_B = torch.nn.L1Loss()(self.fake_B, self.fakeB_static_warp) * lambda_warp
        
        # coherent loss based on warping
        self.fake_B_warp = sparse_image_warp.sparse_image_warp(self.fake_B.permute(0,2,3,1).cuda(self.gpu_p), self.target_B_lm_68[:,:,[1,0]].cuda(self.gpu_p), self.target_B2_lm_68[:,:,[1,0]].cuda(self.gpu_p))[0].permute(0,3,1,2)
        self.loss_warp_inter1 = torch.nn.L1Loss()(self.fake_B2, self.fake_B_warp.detach().cuda(self.gpu)) * lambda_warp_inter
        
        if self.opt.identity_loss == 1:
            # between fakeB and realA
            self.real_A_gray = 0.299*self.real_A[:,0,...] + 0.587*self.real_A[:,1,...] + 0.114*self.real_A[:,2,...]
            self.real_A_rep = self.real_A_gray.repeat(1,3,1,1)
            self.fake_B_rep = self.fake_B.repeat(1,3,1,1)
            self.loss_iden_B = torch.mean(self.faceidenloss(self.fake_B_rep, self.real_A_rep, 
                                        bbox1=self.winB, bbox2=self.winA)) * self.opt.lambda_face
        elif self.opt.identity_loss == 2:
            # between fakeB and fakeB_static
            self.fake_B_rep = self.fake_B.repeat(1,3,1,1)
            self.fakeB_static_rep = self.fakeB_static.repeat(1,3,1,1)
            self.loss_iden_B = torch.mean(self.faceidenloss(self.fake_B_rep.cuda(self.gpu_p), self.fakeB_static_rep.cuda(self.gpu_p), bbox1=self.winB, bbox2=self.winA)).cuda(self.gpu) * self.opt.lambda_face


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A
        
        if getattr(self,'loss_geom_B',-1) != -1:
            self.loss_G = self.loss_G + self.loss_geom_B
        if getattr(self,'loss_geom_B_lipline',-1) != -1:
            self.loss_G = self.loss_G + self.loss_geom_B_lipline
        
        if getattr(self,'loss_warp_B',-1) != -1:
            self.loss_G = self.loss_G + self.loss_warp_B
        
        if getattr(self, 'loss_warp_inter1',-1) != -1:
            self.loss_G = self.loss_G + self.loss_warp_inter1
        
        if getattr(self,'loss_iden_B',-1) != -1:
            self.loss_G = self.loss_G + self.loss_iden_B

        if getattr(self,'loss_G_A_l',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_l
        if getattr(self,'loss_G_A_le',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_le
        if getattr(self,'loss_G_A_ll',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_ll
        if getattr(self,'loss_G_A_coh',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_coh
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A
        self.set_requires_grad([self.netD_A], False)  # Ds require no gradients when optimizing Gs
        if self.opt.use_mask:
            self.set_requires_grad([self.netD_A_l], False)
        if self.opt.use_eye_mask:
            self.set_requires_grad([self.netD_A_le], False)
        if self.opt.use_lip_mask:
            self.set_requires_grad([self.netD_A_ll], False)
        if self.opt.coherent:
            self.set_requires_grad([self.netD_A_coh], False)
        self.optimizer_G.zero_grad()  # set G_A's gradients to zero
        self.backward_G()             # calculate gradients for G_A
        self.optimizer_G.step()       # update G_A's weights
        # D_A
        self.set_requires_grad([self.netD_A], True)
        if self.opt.use_mask:
            self.set_requires_grad([self.netD_A_l], True)
        if self.opt.use_eye_mask:
            self.set_requires_grad([self.netD_A_le], True)
        if self.opt.use_lip_mask:
            self.set_requires_grad([self.netD_A_ll], True)
        if self.opt.coherent:
            self.set_requires_grad([self.netD_A_coh], True)
        self.optimizer_D.zero_grad()   # set D_A's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        if self.opt.use_mask:
            self.backward_D_A_l()# calculate gradients for D_A_l
        if self.opt.use_eye_mask:
            self.backward_D_A_le()# calculate gradients for D_A_le
        if self.opt.use_lip_mask:
            self.backward_D_A_ll()# calculate gradients for D_A_ll
        if self.opt.coherent:
            self.backward_D_A_coh()
        self.optimizer_D.step()  # update D_A's weights