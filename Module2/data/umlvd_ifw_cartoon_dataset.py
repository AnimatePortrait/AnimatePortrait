import os.path
from data.base_dataset import BaseDataset, get_params2, get_transform, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
import cv2
from scipy.interpolate import griddata
from torch.nn import functional as F

def trans_lm(lm, transform_params, opt, size, win=None, rx=0.15, ry=0.2, rs=0.7):
    w, h = size
    tx = transform_params['crop_pos'][0]
    ty = transform_params['crop_pos'][1]
    flip = transform_params['flip']
    #lm: (x,y)
    lm[:,0] = lm[:,0]*opt.load_size/w - tx
    lm[:,1] = lm[:,1]*opt.load_size/h - ty
    if flip and not opt.no_flip:
        lm[:,0] = opt.crop_size - lm[:,0]
        lm[:68,:] = lm[[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65],:]
        if lm.shape[0] > 68:
            lm[68:,:] = lm[[69,68,70,72,71,74,73,75,77,76],:]
    if win is None:
        x1 = int(round(rx*opt.load_size - tx))
        x2 = x1 + int(round(rs*opt.load_size))
        y1 = int(round(ry*opt.load_size - ty))
        y2 = y1 + int(round(rs*opt.load_size))
    else:
        x1,x2,y1,y2 = win
        box_size = int(round((x2-x1)*opt.load_size/w))
        x1 = int(round(x1*opt.load_size/w - tx))
        x2 = x1 + box_size
        y1 = int(round(y1*opt.load_size/h - ty))
        y2 = y1 + box_size
    if flip and not opt.no_flip:
        temp = x1
        x1 = opt.crop_size - x2
        x2 = opt.crop_size - temp
    return lm, torch.IntTensor([x1,x2,y1,y2])

    
def cal_warp256(frame0, lm2d0, lm2d):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    edges = np.array([[0, 0], [255, 255], [0, 255], [255, 0], [0, 255], [255, 0], [255, 255], [255, 255]])
    lm2d = lm2d[:,[1,0]]
    lm2d0 = lm2d0[:,[1,0]]
    destination = np.concatenate((lm2d, edges))
    source = np.concatenate((lm2d0, edges))
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(256,256)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(256,256)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    frame = cv2.remap(frame0, map_x_32, map_y_32, cv2.INTER_LINEAR)
    #frame = cv2.remap(frame0, map_x_32, map_y_32, cv2.INTER_CUBIC)
    return frame
    
def cal_motion256(lm2d0, lm2d):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    edges = np.array([[0, 0], [255, 255], [0, 255], [255, 0], [0, 255], [255, 0], [255, 255], [255, 255]])
    lm2d = lm2d[:,[1,0]]
    lm2d0 = lm2d0[:,[1,0]]
    destination = np.concatenate((lm2d, edges))
    source = np.concatenate((lm2d0, edges))
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(256,256)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(256,256)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    map_xy = np.stack([map_x_32, map_y_32], axis = 2)
    map_xy = map_xy/127.5 - 1
    return map_xy

def txt2lm(path):
    lines = open(path).read().splitlines()
    lm = [[float(e.split()[0]), float(e.split()[1])] for e in lines]
    lm = np.array(lm)
    lm = lm[:,[1,0]]
    return lm

def cal_warp_path(frame0p, lm2d0p, lm2dp):
    frame0 = cv2.imread(frame0p)
    lm2d0 = txt2lm(lm2d0p)
    lm2d = txt2lm(lm2dp)
    grid_x, grid_y = np.mgrid[0:511:512j, 0:511:512j]
    edges = np.array([[0, 0], [511, 511], [0, 511], [511, 0], [0, 255], [255, 0], [255, 511], [511, 255]])
    destination = np.concatenate((lm2d, edges))
    source = np.concatenate((lm2d0, edges))
    grid_z = griddata(destination, source, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(512,512)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(512,512)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    frame = cv2.remap(frame0, map_x_32, map_y_32, cv2.INTER_CUBIC)
    return frame

def transform_mask(mask,dx,dy):
    theta = torch.tensor([
        [1,0,dx],
        [0,1,dy]
        ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), mask.unsqueeze(0).size())
    output = F.grid_sample(mask.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch
    


class UMLVDIFWCartoonDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        imglistA = 'datasets/list/%s/%s.txt' % (opt.phase+'A', opt.dataroot)
        imglistB = 'datasets/list/%s/%s.txt' % (opt.phase+'B', opt.dataroot)
        
        if not os.path.exists(imglistA) or not os.path.exists(imglistB):
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        else:
            self.A_paths = sorted(open(imglistA, 'r').read().splitlines())
            self.B_paths = sorted(open(imglistB, 'r').read().splitlines())
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print("A size:", self.A_size)
        print("B size:", self.B_size)
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs, used for training!
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # realA and realB
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        Alm = Image.open(A_path.replace('/Photo/','/Alm/MTCNN/'))
        Brlm = Image.open(B_path.replace('/Cartoon/','/Blm/MTCNN/'))
        Alm_68 = open(A_path.replace('/Photo/','/Alm_txt/MTCNN/')[:-4]+'.txt').read().splitlines()
        Alm_68 = torch.Tensor([[float(e.split()[0]), float(e.split()[1])] for e in Alm_68])
        Brlm_68 = open(B_path.replace('/Cartoon/','/Blm_txt/MTCNN/')[:-4]+'.txt').read().splitlines()
        Brlm_68 = torch.Tensor([[float(e.split()[0]), float(e.split()[1])] for e in Brlm_68])


        basenA = os.path.basename(A_path)
        basenB = os.path.basename(B_path)
        if self.opt.use_mask:
            A_mask_img = Image.open(A_path.replace('/Photo/','/Amask/nose/'))
            Br_mask_img = Image.open(B_path.replace('/Cartoon/','/Bmask/nose/'))
        if self.opt.use_eye_mask:
            A_maske_img = Image.open(A_path.replace('/Photo/','/Amask/eyes/'))
            Br_maske_img = Image.open(B_path.replace('/Cartoon/','/Bmask/eyes/'))
        if self.opt.use_lip_mask:
            A_maskl_img = Image.open(A_path.replace('/Photo/','/Amask/lips/'))
            Br_maskl_img = Image.open(B_path.replace('/Cartoon/','/Bmask/lips/'))

        # apply image transformation (for realA and realB)
        transform_params_A = get_params2(self.opt, A_img.size)
        transform_params_B = get_params2(self.opt, B_img.size)

        Alm_68, winA = trans_lm(Alm_68, transform_params_A, self.opt, A_img.size)
        Brlm_68, winBr = trans_lm(Brlm_68, transform_params_B, self.opt, B_img.size)
        A = get_transform(self.opt, transform_params_A, grayscale=(self.input_nc == 1))(A_img)
        B = get_transform(self.opt, transform_params_B, grayscale=(self.output_nc == 1))(B_img)
        Alm = get_transform(self.opt, transform_params_A, grayscale=True)(Alm)
        Brlm = get_transform(self.opt, transform_params_B, grayscale=True)(Brlm)
        if self.opt.use_mask:
            A_mask = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_mask_img)
            Br_mask = get_transform_mask(self.opt, transform_params_B, grayscale=1)(Br_mask_img)
        if self.opt.use_eye_mask:
            A_maske = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_maske_img)
            Br_maske = get_transform_mask(self.opt, transform_params_B, grayscale=1)(Br_maske_img)
        if self.opt.use_lip_mask:
            A_maskl = get_transform_mask(self.opt, transform_params_A, grayscale=1)(A_maskl_img)
            Br_maskl = get_transform_mask(self.opt, transform_params_B, grayscale=1)(Br_maskl_img)

        item = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_lm':Alm, 'B_lm':Brlm, 'A_lm_68':Alm_68, 'B_lm_68':Brlm_68}
        
        # target_A_lm same as real_A_lm
        item['tA_lm'] = Alm.clone()
        item['tA_lm_68'] = Alm_68.clone()
        item['winA'] = winA
        item['winBr'] = winBr
        item['image_paths'] = basenA[:-4]+'->'+basenB[:-4]+'.png'
        if self.opt.use_mask:
            item['A_mask'] = A_mask
            item['Br_mask'] = Br_mask
        if self.opt.use_eye_mask:
            item['A_maske'] = A_maske
            item['Br_maske'] = Br_maske
        if self.opt.use_lip_mask:
            item['A_maskl'] = A_maskl
            item['Br_maskl'] = Br_maskl
        
        r = random.random()
        if r <= self.opt.select_noniden_thre:
            # target_B_lm same as real_B_lm
            # target_B2_lm = target_B_lm + offset
            item['tB_lm'] = Brlm.clone()
            item['tB_lm_68'] = Brlm_68.clone()
            offset = torch.rand(item['tB_lm_68'].shape) * self.opt.max_offset
            offset2 = torch.rand([1,2]) * self.opt.max_offset
            offset[48:68,:] = offset2.repeat(20,1)
            if self.opt.max_offset > 3:
                offset = torch.rand([1,2]) * self.opt.max_offset
                dx = - offset[0,0] / self.opt.crop_size
                dy = - offset[0,1] / self.opt.crop_size
                offset = offset.repeat(68,1)
            item['tB2_lm_68'] = item['tB_lm_68'] + offset
            tB2_lm = np.zeros((self.opt.crop_size,self.opt.crop_size))
            radius = 5 if self.opt.crop_size == 512 else 3
            for (x, y) in item['tB2_lm_68'].numpy():
                cv2.circle(tB2_lm, (int(round(x)), int(round(y))), radius, 255, -1)
            tB2_lm = torch.from_numpy(tB2_lm)[None,...].float() / 255. * 2 - 1
            item['tB2_lm'] = tB2_lm
            item['winB'] = winBr.clone()
            # assume the offset between tB_lm and tB2_lm is small, use winB
            item['winB2'] = winBr.clone()
            if self.opt.use_mask:
                item['B_mask'] = Br_mask.clone()
                item['B2_mask'] = Br_mask.clone()
                if self.opt.max_offset > 3:
                    item['B2_mask'] = transform_mask(Br_mask,dx,dy)
            if self.opt.use_eye_mask:
                item['B_maske'] = Br_maske.clone()
                item['B2_maske'] = Br_maske.clone()
                if self.opt.max_offset > 3:
                    item['B2_maske'] = transform_mask(Br_maske,dx,dy)
            if self.opt.use_lip_mask:
                item['B_maskl'] = Br_maskl.clone()
                item['B2_maskl'] = Br_maskl.clone()
                if self.opt.max_offset > 3:
                    item['B2_maskl'] = transform_mask(Br_maskl,dx,dy)
            lmBtp = B_path.replace('/Cartoon/','/Blm_txt/MTCNN/')[:-4]+'.txt'#--targetBlm
        else:
            item['tB_lm'] = Alm.clone()
            item['tB_lm_68'] = Alm_68.clone()
            offset = torch.rand(item['tB_lm_68'].shape) * self.opt.max_offset
            offset2 = torch.rand([1,2]) * self.opt.max_offset
            offset[48:68,:] = offset2.repeat(20,1)
            if self.opt.max_offset > 3:
                offset = torch.rand([1,2]) * self.opt.max_offset
                dx = - offset[0,0] / self.opt.crop_size
                dy = - offset[0,1] / self.opt.crop_size
                offset = offset.repeat(68,1)
            item['tB2_lm_68'] = item['tB_lm_68'] + offset
            tB2_lm = np.zeros((self.opt.crop_size,self.opt.crop_size))
            radius = 5 if self.opt.crop_size == 512 else 3
            for (x, y) in item['tB2_lm_68'].numpy():
                cv2.circle(tB2_lm, (int(round(x)), int(round(y))), radius, 255, -1)
            tB2_lm = torch.from_numpy(tB2_lm)[None,...].float() / 255. * 2 - 1
            item['tB2_lm'] = tB2_lm
            item['winB'] = winA.clone()
            item['winB2'] = winA.clone()
            if self.opt.use_mask:
                item['B_mask'] = A_mask.clone()
                item['B2_mask'] = A_mask.clone()
                if self.opt.max_offset > 3:
                    item['B2_mask'] = transform_mask(A_mask,dx,dy)
            if self.opt.use_eye_mask:
                item['B_maske'] = A_maske.clone()
                item['B2_maske'] = A_maske.clone()
                if self.opt.max_offset > 3:
                    item['B2_mask'] = transform_mask(A_maske,dx,dy)
            if self.opt.use_lip_mask:
                item['B_maskl'] = A_maskl.clone()
                item['B2_maskl'] = A_maskl.clone()
                if self.opt.max_offset > 3:
                    item['B2_mask'] = transform_mask(A_maskl,dx,dy)
            lmBtp = A_path.replace('/Photo/','/Alm_txt/MTCNN/')[:-4]+'.txt'#--targetBlm
        
        if self.opt.coh_use_more:
            index_clip3 = random.randint(0, len(self.B12_paths)-1)
            index_clip4 = random.randint(0, len(self.B12_paths)-1)
            index_frame3 = random.randint(0, len(self.B12_paths[index_clip3])-1)
            index_frame4 = random.randint(0, len(self.B12_paths[index_clip4])-1)
            B3_path = self.B12_paths[index_clip3][index_frame3]
            B4_path = self.B12_paths[index_clip4][index_frame4]
            B3_img = Image.open(B3_path).convert('RGB')
            B4_img = Image.open(B4_path).convert('RGB')
            B3 = get_transform(self.opt, transform_params_B, grayscale=(self.output_nc == 1))(B3_img)
            B4 = get_transform(self.opt, transform_params_B, grayscale=(self.output_nc == 1))(B4_img)
            item['B3'] = B3
            item['B4'] = B4
        
        if self.opt.isTrain and (self.opt.warp_loss == 2 or self.opt.identity_loss == 2):
            # use pytorch warp
            fakeBs = A_path.replace('/Photo/','/fakeB5_static/')
            fakeB_static = Image.open(fakeBs).convert('RGB')
            fakeB_static = get_transform(self.opt, transform_params_A, grayscale=(self.output_nc == 1))(fakeB_static)
            item['fakeB_static'] = fakeB_static
            
        #add warping image
        
        real_A = item['A'].unsqueeze(0)
        
        warp_motion = cal_motion256(item['A_lm_68'], item['tB_lm_68'])
        warp_motion = torch.from_numpy(warp_motion).unsqueeze(0)
        realA_static_warp = F.grid_sample(real_A, warp_motion, align_corners=True)
        
        warp_motion2 = cal_motion256(item['A_lm_68'], item['tB2_lm_68'])
        warp_motion2 = torch.from_numpy(warp_motion2).unsqueeze(0)
        realA_static_warp2 = F.grid_sample(real_A, warp_motion2, align_corners=True)

        
        item['realA_static_warp'] = realA_static_warp.squeeze(0)
        item['realA_static_warp2'] = realA_static_warp2.squeeze(0)
        item['warp_motion'] = warp_motion.squeeze(0)
        item['warp_motion2'] = warp_motion2.squeeze(0)
        

        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
