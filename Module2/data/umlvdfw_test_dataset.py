import os.path
from data.base_dataset import BaseDataset, get_params2, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageDraw
import random
import torch
import numpy as np
import cv2
from scipy.interpolate import griddata
from torch.nn import functional as F

def trans_lm(lm, transform_params, opt, size, rx=0.15, ry=0.2, rs=0.7):
    w, h = size
    tx = transform_params['crop_pos'][0]
    ty = transform_params['crop_pos'][1]
    flip = transform_params['flip']
    #lm: (x,y)
    lm[:,0] = lm[:,0]*opt.load_size/w - tx
    lm[:,1] = lm[:,1]*opt.load_size/h - ty
    if flip and not opt.no_flip:
        lm[:,0] = opt.crop_size - lm[:,0]
        lm[:,:] = lm[[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65],:]
    x1 = int(round(rx*opt.load_size - tx))
    x2 = x1 + int(round(rs*opt.load_size))
    y1 = int(round(ry*opt.load_size - ty))
    y2 = y1 + int(round(rs*opt.load_size))
    if flip and not opt.no_flip:
        temp = x1
        x1 = opt.crop_size - x2
        x2 = opt.crop_size - temp
    return lm, torch.IntTensor([x1,x2,y1,y2])

faceLmarkLookup = np.load('faceLmarkLookup.npy')
def draw2(height, width, lands, radius, thickness, c=None, op=0):
    if op == 0:
        frame = np.zeros((height,width),dtype=np.uint8) * 255
        lands = np.round(lands).astype(int)
        color = c if c is not None else 255
        for (x,y) in lands:
            cv2.circle(frame, (x,y), radius, color, -1)
        frame = torch.from_numpy(frame)[None,...].float() / 255. * 2 - 1
    elif op == 1:
        frame = np.zeros((height,width),dtype=np.uint8) * 255
        lands = np.round(lands).astype(int)
        color = c if c is not None else 255
        for (x,y) in lands:
            cv2.circle(frame, (x,y), radius, color, -1)
        for refpts in faceLmarkLookup:
            start_point = (lands[refpts[0],0],lands[refpts[0],1])
            end_point = (lands[refpts[1],0],lands[refpts[1],1])
            frame = cv2.line(frame, start_point, end_point, color, thickness)
        frame = torch.from_numpy(frame)[None,...].float() / 255. * 2 - 1
    elif op == 2:
        lands = np.round(lands).astype(int)
        img = Image.new('RGB', (width, height), (255,255,255))
        loops = [list(range(17))+[26,25,24,23,22,21,20,19,18,17],[21,20,19,18,17],[26,25,24,23,22],[39,38,37,36,41,40],[45,44,43,42,47,46],[35,27,31,32,33,34],[54,53,52,51,50,49,48,60,61,62,63,64],[54,64,65,66,67,60,48,59,58,57,56,55],[64,63,62,61,60,67,66,65]]
        colors = ['pink','blue','green','purple','cyan','brown','orange','yellow','magenta']
        draw = ImageDraw.Draw(img)
        ind = 0
        for loop in loops:
            l = [(lands[i][0], lands[i][1]) for i in loop]
            draw.polygon(l, fill=colors[ind], outline='black')
            ind += 1
        frame = torch.from_numpy(np.transpose(np.array(img),(2,0,1))).float() / 255. * 2 - 1
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


class UMLVDFWTestDataset(BaseDataset):
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
        A_img = Image.open(A_path).convert('RGB')
        
        Alm_68 = open(A_path.replace('/Photo/','/Alm_txt/MTCNN/')[:-4]+'.txt').read().splitlines()
        Alm_68 = torch.Tensor([[float(e.split()[0]), float(e.split()[1])] for e in Alm_68])
        if 'Alm' in B_path:
            Blm_68 = open(B_path.replace('/Alm/MTCNN/','/Alm_txt/MTCNN/')[:-4]+'.txt').read().splitlines()
        else:
            Blm_68 = open(B_path.replace('/Drawing/','/Blm_txt/MTCNN/')[:-4]+'.txt').read().splitlines()
        Blm_68 = torch.Tensor([[float(e.split()[0]), float(e.split()[1])] for e in Blm_68])


        basenA = os.path.basename(A_path)
        basenB = os.path.basename(B_path)

        # apply image transformation
        Bsize = A_img.size
        transform_params_A = get_params2(self.opt, A_img.size)
        transform_params_B = get_params2(self.opt, Bsize)
        Alm_68, winA = trans_lm(Alm_68, transform_params_A, self.opt, A_img.size)
        Blm_68, winB = trans_lm(Blm_68, transform_params_B, self.opt, Bsize)
        

        A = get_transform(self.opt, transform_params_A, grayscale=(self.input_nc == 1))(A_img)
        
        radius = 5 if self.opt.crop_size == 512 else 3
        thickness = 4 if self.opt.crop_size == 512 else 2
        Alm = draw2(self.opt.crop_size, self.opt.crop_size, Alm_68.numpy(), radius, thickness, op=self.opt.draw_op)
        Blm = draw2(self.opt.crop_size, self.opt.crop_size, Blm_68.numpy(), radius, thickness, op=self.opt.draw_op)

        item = {'A': A, 'A_paths': A_path, 'B_paths': B_path, 'A_lm':Alm, 'B_lm':Blm}
        
        item['A_lm_68'] = Alm_68
        
        item['tB_lm'] = Blm.clone()
        item['tB_lm_68'] = Blm_68.clone()
        item['winB'] = winB
        item['image_paths'] = basenA[:-4]+'->'+basenB[:-4]+'.png'
        
        real_A = item['A'].unsqueeze(0)
        
        warp_motion = cal_motion256(item['A_lm_68'], item['tB_lm_68'])
        warp_motion = torch.from_numpy(warp_motion).unsqueeze(0)
        realA_static_warp = F.grid_sample(real_A, warp_motion, align_corners=True)
        item['realA_static_warp'] = realA_static_warp.squeeze(0)
        item['warp_motion'] = warp_motion.squeeze(0)

        return item

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
