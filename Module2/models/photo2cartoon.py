import os
import numpy as np
import cv2
import math
import tensorflow as tf
from tensorflow.python.platform import gfile
import face_alignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

curPath = os.path.abspath(os.path.dirname(__file__))

class FaceSeg:
    def __init__(self, model_path=os.path.join(curPath, 'seg_model_384.pb')):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._graph = tf.Graph()
        self._sess = tf.Session(config=config, graph=self._graph)

        self.pb_file_path = model_path
        self._restore_from_pb()
        self.input_op = self._sess.graph.get_tensor_by_name('input_1:0')
        self.output_op = self._sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')

    def _restore_from_pb(self):
        with self._sess.as_default():
            with self._graph.as_default():
                with gfile.FastGFile(self.pb_file_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')

    def input_transform(self, image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image_input = (image / 255.)[np.newaxis, :, :, :]
        return image_input

    def output_transform(self, output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        output = self._sess.run(self.output_op, feed_dict={self.input_op: image_input})[0]
        return self.output_transform(output, shape=image.shape[:2])

class FaceDetect:
    def __init__(self, device, detector):
        # landmarks will be detected by face_alignment library. Set device = 'cuda' if use GPU.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector=detector)

    def align(self, image):
        landmarks = self.__get_max_face_landmarks(image)

        if landmarks is None:
            return None

        else:
            return self.__rotate(image, landmarks)

    def __get_max_face_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        if preds is None:
            return None

        elif len(preds) == 1:
            return preds[0]

        else:
            # find max face
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            return preds[max_face_index]

    @staticmethod
    def __rotate(image, landmarks):
        # rotation angle
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

        # image size after rotating
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))

        # translation
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2

        # affine matrix
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                      [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        return image_rotate, landmarks_rotate

class Preprocess:
    def __init__(self, device='cpu', detector='dlib'):
        self.detect = FaceDetect(device, detector)  # device = 'cpu' or 'cuda', detector = 'dlib' or 'sfd'
        self.segment = FaceSeg()

    def process(self, image):
        face_info = self.detect.align(image)
        if face_info is None:
            return None
        image_align, landmarks_align = face_info

        face = self.__crop(image_align, landmarks_align)
        mask = self.segment.get_mask(face)
        return np.dstack((face, mask))

    def process2(self, image):
        mask = self.segment.get_mask(image)
        return np.dstack((image, mask))

    @staticmethod
    def __crop(image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])

        # expand bbox
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)

        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)

        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop

class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.light = light

        self.ConvBlock1 = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(3, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                                       nn.InstanceNorm2d(ngf),
                                       nn.ReLU(True))

        self.HourGlass1 = HourGlass(ngf, ngf)
        self.HourGlass2 = HourGlass(ngf, ngf)

        # Down-Sampling
        self.DownBlock1 = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=0, bias=False),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.DownBlock2 = nn.Sequential(nn.ReflectionPad2d(1),
                                        nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=0, bias=False),
                                        nn.InstanceNorm2d(ngf*4),
                                        nn.ReLU(True))

        # Encoder Bottleneck
        self.EncodeBlock1 = ResnetBlock(ngf*4)
        self.EncodeBlock2 = ResnetBlock(ngf*4)
        self.EncodeBlock3 = ResnetBlock(ngf*4)
        self.EncodeBlock4 = ResnetBlock(ngf*4)

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf*4, 1)
        self.gmp_fc = nn.Linear(ngf*4, 1)
        self.conv1x1 = nn.Conv2d(ngf*8, ngf*4, kernel_size=1, stride=1)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            self.FC = nn.Sequential(nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True))
        else:
            self.FC = nn.Sequential(nn.Linear(img_size//4*img_size//4*ngf*4, ngf*4),
                                    nn.ReLU(True),
                                    nn.Linear(ngf*4, ngf*4),
                                    nn.ReLU(True))

        # Decoder Bottleneck
        self.DecodeBlock1 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock2 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock3 = ResnetSoftAdaLINBlock(ngf*4)
        self.DecodeBlock4 = ResnetSoftAdaLINBlock(ngf*4)

        # Up-Sampling
        self.UpBlock1 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=0, bias=False),
                                      LIN(ngf*2),
                                      nn.ReLU(True))

        self.UpBlock2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                                      LIN(ngf),
                                      nn.ReLU(True))

        self.HourGlass3 = HourGlass(ngf, ngf)
        self.HourGlass4 = HourGlass(ngf, ngf, False)

        self.ConvBlock2 = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=0, bias=False),
                                        nn.Tanh())

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.HourGlass1(x)
        x = self.HourGlass2(x)

        x = self.DownBlock1(x)
        x = self.DownBlock2(x)

        x = self.EncodeBlock1(x)
        content_features1 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock2(x)
        content_features2 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock3(x)
        content_features3 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        x = self.EncodeBlock4(x)
        content_features4 = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            x_ = F.adaptive_avg_pool2d(x, 1)
            style_features = self.FC(x_.view(x_.shape[0], -1))
        else:
            style_features = self.FC(x.view(x.shape[0], -1))

        x = self.DecodeBlock1(x, content_features4, style_features)
        x = self.DecodeBlock2(x, content_features3, style_features)
        x = self.DecodeBlock3(x, content_features2, style_features)
        x = self.DecodeBlock4(x, content_features1, style_features)

        x = self.UpBlock1(x)
        x = self.UpBlock2(x)

        x = self.HourGlass3(x)
        x = self.HourGlass4(x)
        out = self.ConvBlock2(x)

        return out, cam_logit, heatmap

class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_out = dim_out

        self.ConvBlock1 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_in, dim_out//2, kernel_size=3, stride=1, bias=False))

        self.ConvBlock2 = nn.Sequential(nn.InstanceNorm2d(dim_out//2),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//2, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock3 = nn.Sequential(nn.InstanceNorm2d(dim_out//4),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//4, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock4 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        residual = x

        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        out = torch.cat((x1, x2, x3), 1)

        if residual.size(1) != self.dim_out:
            residual = self.ConvBlock4(residual)

        return residual + out

class HourGlass(nn.Module):
    def __init__(self, dim_in, dim_out, use_res=True):
        super(HourGlass, self).__init__()
        self.use_res = use_res

        self.HG = nn.Sequential(HourGlassBlock(dim_in, dim_out),
                                ConvBlock(dim_out, dim_out),
                                nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False),
                                nn.InstanceNorm2d(dim_out),
                                nn.ReLU(True))

        self.Conv1 = nn.Conv2d(dim_out, 3, kernel_size=1, stride=1)

        if self.use_res:
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1)
            self.Conv3 = nn.Conv2d(3, dim_out, kernel_size=1, stride=1)

    def forward(self, x):
        ll = self.HG(x)
        tmp_out = self.Conv1(ll)

        if self.use_res:
            ll = self.Conv2(ll)
            tmp_out_ = self.Conv3(tmp_out)
            return x + ll + tmp_out_

        else:
            return tmp_out


class HourGlassBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(HourGlassBlock, self).__init__()

        self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
        self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)

        self.ConvBlock5 = ConvBlock(dim_out, dim_out)

        self.ConvBlock6 = ConvBlock(dim_out, dim_out)
        self.ConvBlock7 = ConvBlock(dim_out, dim_out)
        self.ConvBlock8 = ConvBlock(dim_out, dim_out)
        self.ConvBlock9 = ConvBlock(dim_out, dim_out)

    def forward(self, x):
        skip1 = self.ConvBlock1_1(x)
        down1 = F.avg_pool2d(x, 2)
        down1 = self.ConvBlock1_2(down1)

        skip2 = self.ConvBlock2_1(down1)
        down2 = F.avg_pool2d(down1, 2)
        down2 = self.ConvBlock2_2(down2)

        skip3 = self.ConvBlock3_1(down2)
        down3 = F.avg_pool2d(down2, 2)
        down3 = self.ConvBlock3_2(down3)

        skip4 = self.ConvBlock4_1(down3)
        down4 = F.avg_pool2d(down3, 2)
        down4 = self.ConvBlock4_2(down4)

        center = self.ConvBlock5(down4)

        up4 = self.ConvBlock6(center)
        up4 = F.upsample(up4, scale_factor=2)
        up4 = skip4 + up4

        up3 = self.ConvBlock7(up4)
        up3 = F.upsample(up3, scale_factor=2)
        up3 = skip3 + up3

        up2 = self.ConvBlock8(up3)
        up2 = F.upsample(up2, scale_factor=2)
        up2 = skip2 + up2

        up1 = self.ConvBlock9(up2)
        up1 = F.upsample(up1, scale_factor=2)
        up1 = skip1 + up1

        return up1

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetSoftAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x

class SoftAdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)

        self.w_gamma = Parameter(torch.zeros(1, num_features))
        self.w_beta = Parameter(torch.zeros(1, num_features))

        self.c_gamma = nn.Sequential(nn.Linear(num_features, num_features),
                                     nn.ReLU(True),
                                     nn.Linear(num_features, num_features))
        self.c_beta = nn.Sequential(nn.Linear(num_features, num_features),
                                    nn.ReLU(True),
                                    nn.Linear(num_features, num_features))
        self.s_gamma = nn.Linear(num_features, num_features)
        self.s_beta = nn.Linear(num_features, num_features)

    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)

        w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out


class adaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class Photo2Cartoon:
    def __init__(self, weights = './models/photo2cartoon_weights.pt'):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        assert os.path.exists(weights), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load(weights, map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img, nocrop=0):
        # face alignment and segmentation
        if not nocrop: # crop
            face_rgba = self.pre.process(img)
        else: # no crop
            face_rgba = self.pre.process2(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon
    
    def inference2(self, img, nocrop=0):
        # face alignment and segmentation
        if not nocrop: # crop
            face_rgba = self.pre.process(img)
        else: # no crop
            face_rgba = self.pre.process2(img)
        if face_rgba is None:
            #print('[Step2: face detect] can not detect face!!!')
            return None
        
        #print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        #mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        #face = (face*mask + (1-mask)*255) / 127.5 - 1
        face = face / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon_img = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon_img = (cartoon_img + 1) * 127.5
        #cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon_img = (cartoon_img).astype(np.uint8)
        cartoon_img = cv2.cvtColor(cartoon_img, cv2.COLOR_RGB2BGR)
        #print('[Step3: photo to cartoon] success!')
        return cartoon, cartoon_img