import sys
import os, glob, shutil
import numpy as np
import cv2
from PIL import Image
import argparse
import torch
import pickle
import face_alignment
from MTCNN import detector

def align_mtcnn(imgpath, alignedpath, rerun=False):
    if os.path.exists(alignedpath) and not rerun:
        print('pass align_mtcnn')
        return
    detector1 = detector()
    img = cv2.imread(imgpath)
    height, width, _ = img.shape
    image = Image.open(imgpath).convert('RGB')
    faces, _ = detector1(image)
    maxs = 0
    for k, face in enumerate(faces): 
        x1, y1, x2, y2, _ = face
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size
        if size > maxs:
            size1 = int(round(size/0.7))
            x11 = int(cx - size1//2)
            x21 = x11 + size1
            y11 = int(cy - (size1*11)//20)
            y21 = y11 + size1
            maxs = size
            cropped2 = np.ones((size1,size1,3),dtype=np.uint8)*255
            cropped2[max(0,y11)-y11:min(y21,height)-y11,max(0,x11)-x11:min(width,x21)-x11] = img[max(0,y11):min(y21,height),max(0,x11):min(width,x21)]
            cropped2_face = cv2.resize(cropped2, (512, 512), interpolation=cv2.INTER_CUBIC)
            os.makedirs(os.path.dirname(alignedpath), exist_ok=True)
            cv2.imwrite(alignedpath,cropped2_face)
    torch.cuda.empty_cache()

def vis_landmark(height, width, shape, linewidth=2):
    img = np.ones((height,width,3),dtype=np.uint8) * 255
    shape = shape.astype('int32')
    linewidth = linewidth * (height // 256)
    radius = (height // 256)
    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)
    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))
    for i in range(68):
        img = cv2.circle(img, (shape[i, 0], shape[i, 1]), radius, (0, 0, 255), -1)
    return img

def getTestList(db, rerun):
    if os.path.exists('Module2/datasets/list/testA/%s.txt'%db) and os.path.exists('Module2/datasets/list/testB/%s.txt'%db) and not rerun:
        print('pass getTestList',db)
        return
    im = os.getcwd() + '/Data/Photo/%s_MTCNN/ori.png'%db
    lms = sorted(glob.glob(os.getcwd() + '/Data/Alm/MTCNN/%s_MTCNN/*.png'%db))
    lms = [e for e in lms if 'ori.png' not in e]
    os.makedirs('Module2/datasets/list/testA/', exist_ok=True)
    os.makedirs('Module2/datasets/list/testB/', exist_ok=True)
    nums = len(lms)
    print(nums)
    with open('Module2/datasets/list/testA/%s.txt'%db, 'w') as f:
        for i in range(nums):
            print(im, file=f)
    with open('Module2/datasets/list/testB/%s.txt'%db, 'w') as f:
        for lm in lms:
            print(lm, file=f)
    print('Module2/datasets/list/testA/%s.txt'%db)
    print('Module2/datasets/list/testB/%s.txt'%db)

def test_gan_new(db,audioname,exp,epoch='200',postfix='fakenew', output_lm=0, rerun=False, fps=25):
    # genlist
    getTestList(db, rerun)
            
    # test setting
    checkpoint_base = exp.split('/')[-1]
    modelname = 'geomcgt_ifw_test'
    netG = 'resnet_9blocks_rcatland32_full_ifw'
    draw_op = 0
    blendbg = '--blendbg 1'
    if 'cartoon' not in exp:
        output_nc = 1
    else:
        output_nc = 3
    
    # test
    srcdir = 'Module2/results/%s/test_%s/%s'%(exp,epoch,db)
    rsthtml = 'Module2/results/%s/test_%s/index%s.html'%(exp,epoch,db)
    command = 'cd Module2/; CUDA_VISIBLE_DEVICES=%s python test.py --dataroot %s --name %s --model %s --netG %s --netg_resb_div 3 --netg_resb_disp 3 --output_nc %d --load_size 256 --crop_size 256 --epoch %s --gpu_ids 0 --num_test 10000 --serial_batches --imagefolder %s --draw_op %d %s'%(device, db, exp, modelname, netG, output_nc, epoch, db, draw_op, blendbg)
    if not os.path.exists(srcdir) or not os.path.exists(rsthtml) or rerun:
        os.system(command)
    
    # frames to video
    tardir = 'output/'+db+'/frames_%s/'%postfix
    os.makedirs(tardir,exist_ok=True)
    for i in range(10000):
        if not os.path.exists(srcdir+'/ori->%05d_fake_B.png'%i):
            break
        shutil.copy(srcdir+'/ori->%05d_fake_B.png'%i,tardir+'/%05d.png'%i)
    if 'fore' in checkpoint_base:
        shutil.copy(srcdir+'/ori->00000_real_A.png','output/'+db+'/photo_fore.png')
        shutil.copy(srcdir+'/ori->00000_fg_mask.png','output/'+db+'/photo_fore_mask.png')
    video_name = 'output/'+db+'/output_%s.mp4'%(postfix)
    os.system('ffmpeg -loglevel panic -framerate {}  -i '.format(fps)+tardir+'/%05d.png -c:v libx264 -y -vf format=yuv420p '+video_name)
    os.system('ffmpeg -loglevel panic -i ' + video_name + ' -i ' + audioname + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov'))
    os.remove(video_name)
    os.system('rm -rf ' + tardir)
    print('output is',video_name.replace('.mp4','.mov'))


parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str)
parser.add_argument('--audio', type=str)
parser.add_argument('--video', type=str)
parser.add_argument('--rerun', type=int, default=0, help='whether rerun')
parser.add_argument('--exp', type=str, default='', help='exp name')
parser.add_argument('--epoch', type=str, default='', help='epoch number')

parser.add_argument('--load_AUTOVC_name', type=str, default='Module1/checkpoints/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='Module1/checkpoints/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='Module1/checkpoints/ckpt_content_branch.pth')
parser.add_argument('--amp_lip_x', type=float, default=2.)
parser.add_argument('--amp_lip_y', type=float, default=2.)
parser.add_argument('--amp_pos', type=float, default=.5)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[])
parser.add_argument('--output_folder', type=str, default='Module1/examples')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)

opt_parser = parser.parse_args()

if opt_parser.video:
    tempdir = "./temp"
    os.system("rm -rf "+tempdir)
    os.system("mkdir "+tempdir) 
    print("from video {} get audio and image.".format(opt_parser.video))
    videoname = os.path.basename(opt_parser.video)
    os.system("cp {} {}".format(opt_parser.video, tempdir))
    os.system("ffmpeg -loglevel panic -y -i {}/{} -r 1 -t 1 {}/%d.png".format(tempdir,videoname,tempdir))
    os.system("ffmpeg -loglevel panic -y -i {}/{} -f wav {}/{}.wav".format(tempdir,videoname,tempdir,videoname.split(".")[0]))
    shutil.copy("{}/{}.wav".format(tempdir, videoname.split(".")[0]),"./Module1/audio/{}.wav".format(videoname.split(".")[0], videoname.split(".")[0]))
    os.system("mv {}/1.png {}/{}.png".format(tempdir,tempdir,videoname.split(".")[0]))
    opt_parser.jpg = "{}/{}.png".format(tempdir,videoname.split(".")[0])
    if not opt_parser.audio:
        opt_parser.audio = "./Module1/audio/{}.wav".format(videoname.split(".")[0])
    elif opt_parser.audio:
        os.system("cp {} {}".format(opt_parser.audio, tempdir))
        opt_parser.audio = "{}/{}".format(tempdir, os.path.basename(opt_parser.audio))
    print("output video name: {}; input audio name: {}; input image name: {}".format(videoname, opt_parser.audio, opt_parser.jpg))


img_path = opt_parser.jpg
imgname = os.path.basename(img_path)
align_path = os.getcwd()+'/Module1/input/'+imgname+'/'+os.path.basename(img_path)[:-4]+'_align.png'
align_mtcnn(img_path, align_path, rerun=opt_parser.rerun)

img = cv2.imread(align_path)
print('reading', align_path)
predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
shapes = predictor.get_landmarks(img)
if (not shapes or len(shapes) != 1):
    print('Cannot detect face landmarks. Exit.')
    exit(-1)
shape_3d = shapes[0]
''' Additional manual adjustment to input face landmarks (slimmer lips and wider eyes) '''
shape_3d[49:54, 1] += 1.
shape_3d[55:60, 1] -= 1.
shape_3d[[37,38,43,44], 1] -=2
shape_3d[[40,41,46,47], 1] +=2
''' normalize face as input to audio branch '''
sys.path.append('Module1')
import util.utils as util
shape_3d, scale, shift = util.norm_input_face(shape_3d)


sys.path.append('Module1/thirdparty/AdaptiveWingLoss')
# audio real data
au_data = []
au_emb = []
au_paths = [opt_parser.audio]
src_dir = 'Module1/examples'
os.makedirs(src_dir, exist_ok=True)
tmp_au_path = '{}/tmp.wav'.format(src_dir)
for au_path in au_paths:
    os.system('ffmpeg -y -loglevel error -i {} -ar 16000 {}'.format(au_path, tmp_au_path))
    shutil.copyfile(tmp_au_path, au_path)
    # au embedding
    from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
    me, ae = get_spk_emb(au_path)
    au_emb.append(me.reshape(-1))
    from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
    print('Processing audio file', au_path)
    c = AutoVC_mel_Convertor(src_dir)

    au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=au_path,
           autovc_model_path=opt_parser.load_AUTOVC_name)
    au_data += au_data_i
if(os.path.isfile(tmp_au_path)):
    os.remove(tmp_au_path)

# landmark fake placeholder
fl_data = []
rot_tran, rot_quat, anchor_t_shape = [], [], []
for au, info in au_data:
    au_length = au.shape[0]
    fl = np.zeros(shape=(au_length, 68 * 3))
    fl_data.append((fl, info))
    rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
    rot_quat.append(np.zeros(shape=(au_length, 4)))
    anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))
gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
pickle_file_names = ['fl', 'au', 'gaze']
dump_datas = [fl_data, au_data, gaze]
for i in range(len(pickle_file_names)):
    pickle_file_name = pickle_file_names[i]
    dump_data = dump_datas[i]
    pickle_file_path = os.path.join(src_dir, 'dump', 'random_val_{}.pickle'.format(pickle_file_name))
    os.makedirs(os.path.join(src_dir, 'dump'), exist_ok=True)
    if os.path.exists(pickle_file_path):
        os.remove(pickle_file_path)
    with open(pickle_file_path, 'wb') as fp:
        pickle.dump(dump_data, fp)


from src.approaches.train_audio2landmark import Audio2landmark_model
model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
model.test(au_emb=au_emb)


fls = ['pred_fls_{}_audio_embed.txt'.format(os.path.splitext(os.path.basename(au_path))[0])]
device = os.environ.get('CUDA_VISIBLE_DEVICES')
from scipy.signal import savgol_filter
for i in range(len(fls)):
    flsi = fls[i]
    fl = np.loadtxt(os.path.join('Module1/examples', fls[i])).reshape((-1, 68,3))
    fl[:, :, 0:2] = -fl[:, :, 0:2]
    fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift
    fl = util.add_naive_eye(fl)
    # additional smooth
    fl = fl.reshape((-1, 204))
    fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
    fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
    fl = fl.reshape((-1, 68, 3))
    os.remove(os.path.join('Module1/examples', fls[i]))

    print('au_paths', i, len(au_paths), len(fls), fls)
    au_path = au_paths[i]
    db = '{}-{}'.format(os.path.splitext(os.path.basename(img_path))[0],os.path.splitext(os.path.basename(au_path))[0])
    tranbdir1 = 'Data/Photo/%s_MTCNN/' % db
    tranbdir2 = 'Data/Alm/MTCNN/%s_MTCNN/' % db
    tranbdir3 = 'Data/Alm_txt/MTCNN/%s_MTCNN/' % db
    lmvisdir2 = 'output/' + db + '/landmark_seq2'
    os.makedirs(tranbdir1, exist_ok=True)
    os.makedirs(tranbdir2, exist_ok=True)
    os.makedirs(tranbdir3, exist_ok=True)
    os.makedirs(lmvisdir2, exist_ok=True)
    shutil.copy(align_path, 'output/' + db + '/photo.png')
    # save tar
    h,w = img.shape[:2]
    lm2d = fl[:,:,:2]
    if h == 256:
        h = w = 512
        lm2d = 2 * lm2d
    lm2s = []
    for k in range(lm2d.shape[0]):
        lm2 = lm2d[k]
        lmimg = np.zeros((h,w))
        with open(tranbdir3 + '/%05d.txt'%k, 'w') as f:
            for (x,y) in lm2:
                print(x, y, file=f)
                cv2.circle(lmimg, (int(round(x)), int(round(y))), 5, 255, -1)
        cv2.imwrite(tranbdir2 + '/%05d.png'%k, lmimg)
        # drawLMvis
        lmvis2 = vis_landmark(h,w,lm2)
        cv2.imwrite(lmvisdir2+'/%05d.png'%k,lmvis2)
        lm2s.append(lm2)
    os.system('ffmpeg -loglevel panic -framerate {}  -i '.format(62.5)+lmvisdir2+'/%05d.png -c:v libx264 -y -vf format=yuv420p '+lmvisdir2+'.mp4')
    os.system('ffmpeg -loglevel panic -i ' + lmvisdir2+'.mp4' + ' -i ' + au_path + ' -vcodec copy  -acodec copy -y  ' + lmvisdir2+'.mov')
    os.remove(lmvisdir2+'.mp4')
    os.system('rm -rf ' + lmvisdir2)
    # save ori
    h,w = img.shape[:2]
    flo = shape_3d
    flo[:, 0:2] = -flo[:, 0:2]
    flo[:, 0:2] = flo[:, 0:2] / scale - shift
    lm2 = flo[:,:2]
    if h == 256:
        h = w = 512
        lm2 = 2*lm2
        frame1 = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(tranbdir1+'/ori.png',frame1)
    else:
        shutil.copy(align_path, tranbdir1+'/ori.png')
    lmimg = np.zeros((h,w))
    im2 = cv2.imread(tranbdir1+'/ori.png')
    with open(tranbdir3+'/ori.txt','w') as f:
        for (x,y) in lm2:
            print(x,y,file=f)
            cv2.circle(im2, (int(round(x)),int(round(y))), 5, (0,0,255), -1)
            cv2.circle(lmimg, (int(round(x)), int(round(y))), 5, 255, -1)
    cv2.imwrite(tranbdir2+'/ori.png',lmimg)
    cv2.imwrite(tranbdir1+'/ori_view.png',im2)

    # test gan
    epoch = '70'
    exp = 'formal/cartoon'
    exp = 'formal/drawing'
    if opt_parser.epoch != '':
        epoch = opt_parser.epoch
    if opt_parser.exp != '':
        exp = opt_parser.exp
    postfix = exp.split('/')[-1] + '--ep' + epoch
    test_gan_new(db, au_path, exp, epoch, postfix, rerun=opt_parser.rerun, fps=62.5)
    print('finish gen', db, ' by exp', exp)
