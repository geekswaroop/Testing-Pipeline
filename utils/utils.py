import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
from model.loss import WeightedBCELoss, FocalLoss, DiceLoss, WeightedMSE

from data.dataset import MitoDataset, MitoSkeletonDataset as DistMitoDataset #, NeuronDataset
from data.utils import collate_fn, collate_fn_test

# tensorboardX
from tensorboardX import SummaryWriter

def get_args_train():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    # parser.add_argument('-v','--val',  default='',
    #                     help='input folder (test)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                        help='Ground-truth label path')
    parser.add_argument('-dw','--dis-name', default='im_uint8.h5',
                        help='Loss weight based on distance transform.')

    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='I/O size of deep network')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')                    
    parser.add_argument('-ft','--finetune', type=bool, default=False,
                        help='Fine-tune on previous model [Default: False]')
    parser.add_argument('-pm','--pre-model', type=str, default='',
                        help='Pre-trained model path')                  

    # optimization option
    parser.add_argument('-lt', '--loss', type=int, default=1,
                        help='Loss function')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--volume-total', type=int, default=1000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')
    args = parser.parse_args()
    return args

def get_args_test():
    parser = argparse.ArgumentParser(description='Testing Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='input folder (train)')
    parser.add_argument('-v','--val',  default='',
                        help='input folder (test)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='image data')
    parser.add_argument('-o','--output', default='result/train/',
                        help='output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204')

    # machine option
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='batch size')
    parser.add_argument('-m','--model', help='model used for test')
    parser.add_argument('--file-type', type=str, default='vol')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')

    args = parser.parse_args()
    return args

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_io_size, device

def get_input_train(args, model_io_size, opt='train'):
    # two dataLoader, can't be both multiple-cpu (pytorch issue)

    if opt=='train':
        dir_name = args.train.split('@')
        num_worker = args.num_cpu
        img_name = args.img_name.split('@')
        seg_name = args.seg_name.split('@')
        dis_name = args.dis_name.split('@')
        #dis_name = args.distance.split('@')
    else:
        dir_name = args.val.split('@')
        num_worker = 1
        img_name = args.img_name_val.split('@')
        seg_name = args.seg_name_val.split('@')
        dis_name = args.dis_name.split('@')

    print(img_name)
    print(seg_name)
    print(dis_name)

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    seg_name = [dir_name[0] + x for x in seg_name]
    img_name = [dir_name[0] + x for x in img_name]
    dis_name = [dir_name[0] + x for x in dis_name]
    #dis_name = [x for x in dis_name]
    #print(dis_name)
    #print(img_name)
    #print(seg_name)
    
    # 1. load data
    assert len(img_name)==len(seg_name)
    train_input = [None]*len(img_name)
    train_label = [None]*len(seg_name)
    train_distance = [None]*len(dis_name)
    #train_distance = [None]*len(dis_name)

    # original image is in [0, 255], normalize to [0, 1]
    for i in range(len(img_name)):
        # train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        # train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        # train_input[i] = (train_input[i].transpose(2, 1, 0))[100:-100, 200:-200, 200:-200]
        # train_label[i] = (train_label[i].transpose(2, 1, 0))[100:-100, 200:-200, 200:-200]
        train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        train_distance[i] = np.array(h5py.File(dis_name[i], 'r')['main'])
        #train_distance[i] = np.array(h5py.File(dis_name[i], 'r')['main'])[14:-14, 200:-200, 200:-200]

        # downsample
        # from skimage.measure import block_reduce
        # from skimage.transform import resize
        # train_label[i] = block_reduce(train_label[i], (1,2,2), np.max)
        # train_input[i] = resize(train_input[i], train_label[i].shape, mode='reflect',
        #             order=3, anti_aliasing=True, anti_aliasing_sigma=(0, 0.25, 0.25))

        train_label[i] = (train_label[i] != 0).astype(np.float32)
        train_input[i] = train_input[i].astype(np.float32)
        train_distance[i] = train_distance[i].astype(np.float32)
        #train_distance[i] = train_distance[i].astype(np.float32)
    
        #train_input[i] = ndimage.zoom(train_input[i], [1, 0.5, 0.5], mode='nearest')

        #print("maximum distance: ", np.amax(train_distance[i]))
        assert train_input[i].shape==train_label[i].shape
        # assert train_input[i].shape==train_distance[i].shape
        # print("input type: ", train_input[i].dtype)
        print("foreground pixels: ", np.sum(train_label[i]))
        print("volume shape: ", train_input[i].shape)    

    data_aug = True
    print('Data augmentation: ', data_aug)

    # get border
    # print('use border weight:')
    # border_file = '/n/coxfs01/zudilin/research/mitoNet/data/file/Lucchi/train_border.h5'
    # train_border = np.array(h5py.File(border_file, 'r')['main'])/255.0 
    # train_border = train_border.astype(np.float32)
    # print(border_file)

    # print('use distance transform:')
    # distance_file = '/n/coxfs01/zudilin/research/mitoNet/data/file/Lucchi/skeleton_new_0.h5'
    # train_distance = np.array(h5py.File(distance_file, 'r')['main'])
    # train_distance = train_distance.astype(np.float32)
    # print(distance_file)

    # print('Data augmentation: ', data_aug)
    # dataset = SynapseDataset(volume=train_input, label=train_label, vol_input_size=model_io_size, \
    #                              vol_label_size=model_io_size, data_aug = data_aug, mode = 'train')
    # dataset = MitoDataset(volume=train_input, label=train_label, border=[train_border], vol_input_size=model_io_size,
    #                 vol_label_size=model_io_size, data_aug = data_aug, mode = 'train') 
    # dataset = NeuronDataset(volume=train_input, label=train_label, border=None, vol_input_size=model_io_size,
    #                 vol_label_size=model_io_size, data_aug = data_aug, mode = 'train', distance=train_distance,
    #                 connect_loss=False)
    dataset = DistMitoDataset(volume=train_input, label=train_label, border=None, vol_input_size=model_io_size,
                    vol_label_size=model_io_size, data_aug = data_aug, mode = 'train', distance=train_distance)                                            
    # to have evaluation during training (two dataloader), has to set num_worker=0
    SHUFFLE = (opt=='train')
    print('Mini-batch size: ', args.batch_size)
    img_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
            num_workers=num_worker, pin_memory=True)
    return img_loader

def get_input_test(args, model_io_size, opt='test'):
    # two dataLoader, can't be both multiple-cpu (pytorch issue)
    file_type = args.file_type
    print(args.file_type)
    assert (file_type in ['vol', 'img'])
    dir_name = args.train.split('@')
    img_name = args.img_name.split('@')

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    img_name = [dir_name[0] + x for x in img_name]
    print(img_name)
    # 1. load data
    print('number of volumes:', len(img_name))
    test_input = [None]*len(img_name)
    result = [None]*len(img_name)
    weight = [None]*len(img_name)

    # original image is in [0, 255], normalize to [0, 1]
    #pad_sz = model_io_size[1] // 2 + 32
    pad_sz = 8
    print('padding size: ', pad_sz)
    for i in range(len(img_name)):
        if file_type == 'vol':
            test_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
            print("volume shape: ", test_input[i].shape)
            # padding
            test_input[i] = np.pad(test_input[i], ((0,0), (pad_sz,pad_sz), (pad_sz,pad_sz)), 'reflect')
            # test_input[i] = test_input[i].transpose(0,2,1)
            # test_input[i] = test_input[i][::-1]
            # test_input[i] = (test_input[i].transpose(2, 1, 0))
            # test_input[i] = np.transpose(test_input[i], (2,0,1))
        elif file_type == 'img':  
            test_input[i] = np.array(imageio.imread(img_name[i]))/255.0
            test_input[i] = np.expand_dims(test_input[i], axis=0)
            print("volume shape: ", test_input[i].shape)
            # padding
            test_input[i] = np.pad(test_input[i], ((0,0), (pad_sz,pad_sz), (pad_sz,pad_sz)), 'reflect')
            # from skimage.measure import block_reduce
            # from skimage.transform import resize
            # temp = block_reduce(test_input[i], (1,2,2), np.max)
            # test_input[i] = resize(test_input[i], temp.shape, mode='reflect',
            #             order=3, anti_aliasing=True, anti_aliasing_sigma=(0, 0.25, 0.25))

        print("volume shape: ", test_input[i].shape)
        result[i] = np.zeros(test_input[i].shape)
        weight[i] = np.zeros(test_input[i].shape)

    dataset = MitoDataset(volume=test_input, label=None, sample_input_size=model_io_size, \
                             sample_label_size=None, sample_stride=model_io_size/2, \
                             augmentor=None, mode='test')
    # to have evaluation during training (two dataloader), has to set num_worker=0
    SHUFFLE = (opt=='train')
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn_test,
            num_workers=args.num_cpu, pin_memory=True)
    return img_loader, result, weight

def gaussian_blend(sz):  
    zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                                np.linspace(-1,1,sz[1], dtype=np.float32),
                                np.linspace(-1,1,sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    sigma, mu = 0.5, 0.0
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    print('weight shape:', ww.shape)
    return ww

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def adjust_lr(args, optimizer, iteration, factor):
    init_lr = args.lr
    lr = init_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    print('Learning rate is %f from iteration %d' % (lr, iteration))        

def setup_loss(loss_type):
    if loss_type == 1:
        criterion = WeightedBCELoss()
    elif loss_type == 2:    
        criterion = WeightedMSE()
    return criterion  

def test(args, test_loader, result, weight, model, device, model_io_size, out_num=5):
    # switch to train mode
    # model.train()
    model.eval()
    volume_id = 0
    ww = gaussian_blend(model_io_size)
    print('out_num: ', out_num)
    if out_num==5:
        skeleton = [x.copy() for x in result]

    start = time.time()
    with torch.no_grad():
        for i, (pos, volume) in enumerate(test_loader):
            volume_id += args.batch_size
            print('volume_id:', volume_id)

            # for gpu computing
            volume = volume.squeeze()
            if len(volume.size())>=3: volume = volume.unsqueeze(1)
            elif len(volume.size())==2: volume = volume.unsqueeze(0).unsqueeze(0)
            if i==0: print('volume size:', volume.size())

            volume = volume.to(device)
            if out_num==4:
                output, _, _, _ = model(volume)
            elif out_num==5:
                output, so0, _, _, _ = model(volume)  
            else:
                print('Invalid output number!')
                exit(0)    

            # sz = model_io_size
            # if i==0: 
            #     print("volume size:", volume.size())
            #     print("output size:", output.size())
            #     hk = h5py.File(args.output+'/demo.h5','w')
            #     data = output[0].cpu().detach().squeeze().numpy().reshape(sz)
            #     print(np.max(data))
            #     print(np.min(data))
            #     hk.create_dataset('main', data=data, compression='gzip')
            #     hk.close()

            sz = model_io_size
            for idx in range(output.size()[0]):
                st = pos[idx]
                result[st[0]][st[1]:st[1]+sz[0], st[2]:st[2]+sz[1], \
                st[3]:st[3]+sz[2]] += output[idx].cpu().detach().squeeze().numpy().reshape(sz) * ww

                if out_num==5:
                    skeleton[st[0]][st[1]:st[1]+sz[0], st[2]:st[2]+sz[1], \
                    st[3]:st[3]+sz[2]] += so0[idx].cpu().detach().squeeze().numpy().reshape(sz) * ww

                weight[st[0]][st[1]:st[1]+sz[0], st[2]:st[2]+sz[1], \
                st[3]:st[3]+sz[2]] += ww

    end = time.time()
    print("prediction time:", (end-start))

    pad_sz = 8
    print('padding size: ', pad_sz)
    for vol_id in range(len(result)):
        result[vol_id] = result[vol_id] / weight[vol_id]
        data = result[vol_id]
        if out_num==5: skel = skeleton[vol_id]
        # data = (result[vol_id]*255).astype(np.uint8)
        if args.file_type == 'vol':
            print('save energy map:')
            #data = data[2:-2]
            data = data[:, pad_sz:-pad_sz, pad_sz:-pad_sz]
            data = (data + 1.0)/2.0 # data is in [-1,1]
            data = (data*255).astype(np.uint8)
            print("volume shape: ", data.shape)
            hf = h5py.File(args.output+'/energy_'+str(vol_id)+'.h5','w')
            hf.create_dataset('main', data=data, compression='gzip')
            hf.close()

            if out_num==5:
                print('save skeleton map:')
                skel = skel[4:-4, pad_sz:-pad_sz, pad_sz:-pad_sz]
                #skel = (skel + 1.0)/2.0 skel is in [0, 1]
                skel = (skel*255).astype(np.uint8)
                print("volume shape: ", data.shape)
                hk = h5py.File(args.output+'/skeleton_'+str(vol_id)+'.h5','w')
                hk.create_dataset('main', data=skel, compression='gzip')
                hk.close()

        elif args.file_type == 'img':   
            data = data[:, pad_sz:-pad_sz, pad_sz:-pad_sz]
            data = (data + 1.0)/2.0
            data = (data*255).astype(np.uint8)
            data = np.squeeze(data)
            print("image size: ", data.shape)
            imageio.imsave(args.output+'/'+args.img_name.split('@')[0], data)

def visualize(model, volume, label, output, output_side, iteration, writer):
    ###
    sz_so = output_side.size()
    so_visual = output_side.detach().cpu().expand(sz_so[0],3,sz_so[2],sz_so[3])
    so_show = vutils.make_grid(so_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Side output', so_show, iteration)

    temp = label.clone()
    temp[(label<0.99)] = 0.0
    label_so = F.max_pool2d(temp, kernel_size=(2,2), stride=(2,2))
    so_visual = label_so.detach().cpu().expand(sz_so[0],3,sz_so[2],sz_so[3])
    so_show = vutils.make_grid(so_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Side label', so_show, iteration)
    ###

    sz = volume.size()
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])

    volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
    output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Input', volume_show, iteration)
    writer.add_image('Label', label_show, iteration)
    writer.add_image('Output', output_show, iteration)    

def visualize_so3(model, volume, label, output, output_so1, output_so2, output_so3, iteration, writer):
    ###
    sz_so1 = output_so1.size()
    so1_visual = output_so1.detach().cpu().expand(sz_so1[0],3,sz_so1[2],sz_so1[3])
    so1_show = vutils.make_grid(so1_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('SO1', so1_show, iteration)

    sz_so2 = output_so2.size()
    so2_visual = output_so2.detach().cpu().expand(sz_so2[0],3,sz_so2[2],sz_so2[3])
    so2_show = vutils.make_grid(so2_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('SO2', so2_show, iteration)

    sz_so3 = output_so3.size()
    so3_visual = output_so3.detach().cpu().expand(sz_so3[0],3,sz_so3[2],sz_so3[3])
    so3_show = vutils.make_grid(so3_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('SO3', so3_show, iteration)

    temp = label.clone()
    temp[(label<0.99)] = 0.0
    label_so = F.max_pool2d(temp, kernel_size=(2,2), stride=(2,2))
    sz_so = label_so.size()
    so_visual = label_so.detach().cpu().expand(sz_so[0],3,sz_so[2],sz_so[3])
    so_show = vutils.make_grid(so_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Side label', so_show, iteration)

    ###
    sz = volume.size()
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])

    volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
    output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Input', volume_show, iteration)
    writer.add_image('Label', label_show, iteration)
    writer.add_image('Output', output_show, iteration)                 


def visualize_so4(model, volume, label, output, 
                  output_so0, output_so1, output_so2, output_so3, 
                  iteration, writer, image_id=None, mask=None,
                  val_img=None, val_out=None):
    ###
    if output_so0 is not None:
        sz_so0 = output_so0.size()
        so0_visual = output_so0.detach().cpu().expand(sz_so0[0],3,sz_so0[2],sz_so0[3])
        so0_show = vutils.make_grid(so0_visual, nrow=8, normalize=True, scale_each=True)
        writer.add_image('SO0', so0_show, iteration)

    if output_so1 is not None:
        sz_so1 = output_so1.size()
        so1_visual = output_so1.detach().cpu().expand(sz_so1[0],3,sz_so1[2],sz_so1[3])
        so1_show = vutils.make_grid(so1_visual, nrow=8, normalize=True, scale_each=True)
        writer.add_image('SO1', so1_show, iteration)

    if output_so2 is not None:
        sz_so2 = output_so2.size()
        so2_visual = output_so2.detach().cpu().expand(sz_so2[0],3,sz_so2[2],sz_so2[3])
        so2_show = vutils.make_grid(so2_visual, nrow=8, normalize=True, scale_each=True)
        writer.add_image('SO2', so2_show, iteration)

    if output_so3 is not None:
        sz_so3 = output_so3.size()
        so3_visual = output_so3.detach().cpu().expand(sz_so3[0],3,sz_so3[2],sz_so3[3])
        so3_show = vutils.make_grid(so3_visual, nrow=8, normalize=True, scale_each=True)
        writer.add_image('SO3', so3_show, iteration)

    temp = label.clone()
    temp[(label<0.99)] = 0.0
    label_so = temp.clone()
    #label_so = F.max_pool2d(temp, kernel_size=(2,2), stride=(2,2))
    sz_so = label_so.size()
    so_visual = label_so.detach().cpu().expand(sz_so[0],3,sz_so[2],sz_so[3])
    so_show = vutils.make_grid(so_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('SOl', so_show, iteration)

    ###
    sz = volume.size()
    volume_visual = volume.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    output_visual = output.detach().cpu().expand(sz[0],3,sz[2],sz[3])
    label_visual = label.detach().cpu().expand(sz[0],3,sz[2],sz[3])

    volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
    output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)
    writer.add_image('Input', volume_show, iteration)
    writer.add_image('Label', label_show, iteration)
    writer.add_image('Output', output_show, iteration) 

    if image_id is not None:
        id_name = ','.join([str(x) for x in image_id])
        writer.add_text('Image ID', id_name, iteration)

    if mask is not None:
        sz = mask.size()
        mask_visual = mask.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        mask_show = vutils.make_grid(mask_visual, nrow=8, normalize=True, scale_each=True)
        #mask_show = (mask_show*255).astype(np.uint8)
        writer.add_image('Weight', mask_show, iteration)

    if (val_img is not None) and (val_out is not None):
        sz = val_img.size()
        val_img_visual = val_img.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        val_out_visual = val_out.detach().cpu().expand(sz[0],3,sz[2],sz[3])
        val_img_show = vutils.make_grid(val_img_visual, nrow=8, normalize=True, scale_each=True)
        val_out_show = vutils.make_grid(val_out_visual, nrow=8, normalize=True, scale_each=True)
        writer.add_image('Val_img', val_img_show, iteration)
        writer.add_image('Val_out', val_out_show, iteration) 