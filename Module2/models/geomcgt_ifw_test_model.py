import torch
from .base_model import BaseModel
from . import networks
import os
import numpy as np

import json
import argparse
from torch.nn import functional as F
from .photo2cartoon import *

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

def cal_warp_frame(frame0, lm2d0, lm2d, edge=True):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    edges = np.array([[0, 0], [255, 255], [0, 255], [255, 0], [0, 127], [127, 0], [127, 255], [255, 127]])
    lm2d0 = lm2d0[0].cpu().data.numpy()
    lm2d = lm2d[0].cpu().data.numpy()
    if edge:
        destination = np.concatenate((lm2d, edges))
        source = np.concatenate((lm2d0, edges))
    else:
        destination = lm2d.copy()
        source = lm2d0.copy()
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(256,256)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(256,256)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    map_xy = np.stack([map_x_32, map_y_32], axis = 2)
    map_xy = map_xy/127.5 - 1
    warp_motion = torch.from_numpy(map_xy).cuda().unsqueeze(0)
    frame = F.grid_sample(frame0, warp_motion, align_corners=True)
    return frame

class GeomCGTIFWTestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--netg_resb_div', type=int, default=3, help='div')
        parser.add_argument('--netg_resb_disp', type=int, default=1, help='disp')
        parser.add_argument('--truncate', type=float, default=0.0, help='whether truncate in forward')
        parser.set_defaults(dataset_mode='umlvdfw_test')
        parser.add_argument('--draw_op', type=int, default=0, help='use which format to draw landmark')
        # blend bg
        parser.add_argument('--blendbg', type=int, default=0, help='whether blend with bg')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'real_A_lm', 'target_B_lm', 'fake_B', 'fake_B_vis']
        visual_names_A.append('fg_mask')
        visual_names_A.append('fakeB_static')
        visual_names_A.append('fake_B_fore')
        visual_names_A.append('fg_mask1')
        
        self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        div=self.opt.netg_resb_div, disp=self.opt.netg_resb_disp)
        
        self.visual_names = visual_names_A
        print(self.visual_names)

        self.netF = load_flow_network()
        self.netF.eval()
        self.netF.cuda()

        from .modnet import MODNet
        self.gpu = self.opt.gpu_ids[0]
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = torch.nn.DataParallel(self.modnet).cuda(self.gpu)
        self.modnet.load_state_dict(torch.load('checkpoints/modnet_photographic_portrait_matting.ckpt'))
        self.modnet.eval()
        if 'drawing' in self.opt.name:
            self.net_staticG = networks.define_G(3, 1, 64, "resnet_style2_9blocks", "instance", use_dropout=False, gpu_ids=self.opt.gpu_ids)
            self.net_staticG.module.load_state_dict(torch.load('checkpoints/static/drawing.pth'))
            self.net_staticG.eval()
        elif 'cartoon' in self.opt.name:
            self.c2p = Photo2Cartoon('checkpoints/static/cartoon.pt')
            
    
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


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.warp_motion = input['warp_motion'].to(self.device)
        self.realA_static_warp = input['realA_static_warp'].to(self.device)
        self.real_A_lm = input['A_lm'].to(self.device)
        self.real_A_lm_68 = input['A_lm_68'].to(self.device) 
        self.target_B_lm = input['tB_lm'].to(self.device)
        self.target_B_lm_68 = input['tB_lm_68'].to(self.device)
        self.winB = input['winB'].to(self.device)
        self.image_paths = input['image_paths']
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']
        self.iw_flow, self.real_A_if_mask = flow_network_warp(self.netF, self.real_A, self.real_A_lm_68[:,:68], self.target_B_lm_68[:,:68])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        _, _, matte = self.modnet(self.real_A, True)
        mask = (matte > 0.5).float()
        style_B = torch.Tensor([0., 1., 0.]).view(1, 3, 1, 1).repeat(1, 1, 128, 128).to(self.device)
        if 'drawing' in self.opt.name:
            real_A_512 =  torch.nn.functional.interpolate(input=self.real_A, size=(512, 512), mode='bilinear', align_corners=False)
            fakeB_static_512 = self.net_staticG(real_A_512, style_B)
            self.fakeB_static = torch.nn.functional.interpolate(input=fakeB_static_512, size=(256, 256), mode='bilinear', align_corners=False)
        elif 'cartoon' in self.opt.name:
            inputA = np.transpose(self.real_A[0].cpu().numpy(), (1, 2, 0))
            inputA = (inputA + 1) * 127.5
            inputA = inputA.astype(np.uint8)
            output2, _ = self.c2p.inference2(inputA, nocrop=1)
            self.fakeB_static = output2.unsqueeze(0)
        self.real_A = ((self.real_A/2+0.5)*mask + 1-mask)*2-1
        self.fg_mask = (mask * 2 - 1).repeat(1, 3, 1, 1)
        
        self.fake_B = self.netG_A(self.real_A, self.real_A_lm, self.target_B_lm, self.warp_motion, self.iw_flow, self.real_A_if_mask)
        
        self.mask1 = F.grid_sample(mask, self.warp_motion, align_corners=True)
        self.fake_B_fore = self.fake_B.clone()
        self.fake_B = ((self.fake_B/2+0.5)*self.mask1 + (self.fakeB_static/2+0.5)*(1-self.mask1))*2-1
        self.fg_mask1 = (self.mask1 * 2 - 1).repeat(1, 3, 1, 1)
            
        self.fake_B_vis = self.get_lmvis(self.fake_B, self.target_B_lm_68, self.winB)
    