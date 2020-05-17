import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
from model.loss import WeightedBCELoss, FocalLoss, DiceLoss, WeightedMSE

from data.dataset import MitoDataset, MitoSkeletonDataset as DistMitoDataset
from data.utils import collate_fn, collate_fn_test
from .utils import *

# tensorboardX
from tensorboardX import SummaryWriter

def train(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):
        # if volume_id == 0:
        #     hk = h5py.File(args.output+'/demo_label.h5','w')
        #     sz = (4, 320, 320)
        #     data = label[0].cpu().detach().squeeze().numpy().reshape(sz)
        #     hk.create_dataset('main', data=data, compression='gzip')
        #     hk.close()
        #     exit(0)
        # print('Iteration: ', iteration)
        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size

        # if i == 0: print(volume.size())
        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume = volume.squeeze().unsqueeze(1)
        if iteration==0: print(volume.size())
        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)
        if iteration==0: print(output.size())

        #print(torch.max(output))
        #print(torch.min(output))
        #assert (args.loss == 3 or args.loss == 4)
        loss = criterion(output, label, class_weight)
        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #     

def train_so(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):
        # if volume_id == 0:
        #     hk = h5py.File(args.output+'/demo_label.h5','w')
        #     sz = (4, 320, 320)
        #     data = label[0].cpu().detach().squeeze().numpy().reshape(sz)
        #     hk.create_dataset('main', data=data, compression='gzip')
        #     hk.close()
        #     exit(0)
        # print('Iteration: ', iteration)
        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size

        # if i == 0: print(volume.size())
        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume = volume.squeeze().unsqueeze(1)
        if iteration==0: print(volume.size())

        # prepare label for side-outputs
        temp = label.clone()
        temp[(label<0.99)] = 0.0
        label_so = F.max_pool2d(temp, kernel_size=(2,2), stride=(2,2))
        weight_so = 1.0 + label_so*9
        criterion_so = WeightedBCELoss()

        label_so, weight_so = label_so.to(device), weight_so.to(device)

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output, output_so = model(volume)
        if iteration==0: print(output.size())

        #print(torch.max(output))
        #print(torch.min(output))
        #assert (args.loss == 3 or args.loss == 4)
        loss0 = criterion(output, label, class_weight)
        loss1 = criterion_so(output_so, label_so, weight_so)
        loss = loss0 + 0.01 * loss1
        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 100 == 0:
            #model.eval()
            visualize(model, volume, label, output, output_so, iteration, writer)
            #model.train()

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    # 

def prepare_so(label, device):
    temp = label.clone()
    temp[(label<0.99)] = 0.0

    label_so0 = temp.clone()
    weight_so0 = 1.0 + label_so0*8

    label_so1 = F.max_pool2d(label_so0, kernel_size=(2,2), stride=(2,2))
    weight_so1 = 1.0 + label_so1*8
    
    label_so2 = F.max_pool2d(label_so1, kernel_size=(2,2), stride=(2,2))
    weight_so2 = 1.0 + label_so2*4

    label_so3 = F.max_pool2d(label_so2, kernel_size=(2,2), stride=(2,2))
    weight_so3 = 1.0 + label_so3*2

    label_so0, weight_so0 = label_so0.to(device), weight_so0.to(device)
    label_so1, weight_so1 = label_so1.to(device), weight_so1.to(device)
    label_so2, weight_so2 = label_so2.to(device), weight_so2.to(device)
    label_so3, weight_so3 = label_so3.to(device), weight_so3.to(device)

    return label_so0, label_so1, label_so2, label_so3, weight_so0, weight_so1, weight_so2, weight_so3

def prepare_so3(label, device):
    temp = label.clone()
    temp[(label<0.99)] = 0.0

    label_so0 = temp.clone()
    weight_so0 = 1.0 + label_so0*8

    label_so1 = F.max_pool2d(label_so0, kernel_size=(2,2), stride=(2,2))
    weight_so1 = 1.0 + label_so1*8
    
    label_so2 = F.max_pool2d(label_so1, kernel_size=(2,2), stride=(2,2))
    weight_so2 = 1.0 + label_so2*4

    label_so3 = F.max_pool2d(label_so2, kernel_size=(2,2), stride=(2,2))
    weight_so3 = 1.0 + label_so3*2

    label_so0, weight_so0 = label_so0.to(device), weight_so0.to(device)
    label_so1, weight_so1 = label_so1.to(device), weight_so1.to(device)
    label_so2, weight_so2 = label_so2.to(device), weight_so2.to(device)
    label_so3, weight_so3 = label_so3.to(device), weight_so3.to(device)

    return label_so0, label_so1, label_so2, label_so3, weight_so0, weight_so1, weight_so2, weight_so3

def train_so3(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):
        # if volume_id == 0:
        #     hk = h5py.File(args.output+'/demo_label.h5','w')
        #     sz = (4, 320, 320)
        #     data = label[0].cpu().detach().squeeze().numpy().reshape(sz)
        #     hk.create_dataset('main', data=data, compression='gzip')
        #     hk.close()
        #     exit(0)
        # print('Iteration: ', iteration)
        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size

        # if i == 0: print(volume.size())
        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume = volume.squeeze().unsqueeze(1)
        if iteration==0: print(volume.size())

        criterion_so = WeightedBCELoss()
        # prepare label for side-outputs
        label_so0, label_so1, label_so2, label_so3, weight_so0, weight_so1, weight_so2, weight_so3 = prepare_so3(label, device)
        if iteration==0: print(label_so1.size(), label_so2.size(), label_so3.size())

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output, output_so1, output_so2, output_so3 = model(volume)
        if iteration==0: print(output.size(), output_so1.size(), output_so2.size(), output_so3.size())

        #print(torch.max(output))
        #print(torch.min(output))
        #assert (args.loss == 3 or args.loss == 4)
        loss0 = criterion(output, label, class_weight)
        loss1 = criterion_so(output_so1, label_so1, weight_so1)
        loss2 = criterion_so(output_so2, label_so2, weight_so2)
        loss3 = criterion_so(output_so3, label_so3, weight_so3)

        loss = loss0 + 0.08 * loss1 + 0.04 * loss2 + 0.02 * loss3
        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 100 == 0:
            #model.eval()
            visualize_so3(model, volume, label, output, output_so1, output_so2, output_so3, iteration, writer)
            #model.train()

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #     

def train_so3_ref(args, train_loader, model, device, criterion, optimizer, logger, writer, K=1):
    # switch to train mode
    print('Num of model steps: ', K)
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):

        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size


        volume = volume.squeeze().unsqueeze(1)
        latent = torch.zeros(volume.size(), dtype=volume.dtype)
        if iteration==0: 
            print(volume.size())
            print(latent.size())

        criterion_so = WeightedBCELoss()
        # prepare label for side-outputs
        label_so1, label_so2, label_so3, weight_so1, weight_so2, weight_so3 = prepare_so(label, device)

        volume, label = volume.to(device), label.to(device)
        latent = latent.to(device)
        class_weight = class_weight.to(device)

        loss = torch.tensor(0.0, dtype=volume.dtype).to(device)
        for k in range(K):
            inputv = torch.stack([volume, latent], dim=1).squeeze()
            latent.cpu().detach()
            if iteration==0: print(k, inputv.size())
            output, output_so1, output_so2, output_so3 = model(inputv)
            latent = output.clone()
            latent.detach()
            inputv.cpu().detach()

            loss0 = criterion(output, label, class_weight)
            loss1 = criterion_so(output_so1, label_so1, weight_so1)
            loss2 = criterion_so(output_so2, label_so2, weight_so2)
            loss3 = criterion_so(output_so3, label_so3, weight_so3)

            output.cpu().detach()
            output_so1.cpu().detach()
            output_so2.cpu().detach()
            output_so3.cpu().detach()
            #if iteration==0: print(loss0, loss1, loss2, loss3)
            alpha = 0.05
            beta = torch.tensor((k+1), dtype=volume.dtype).to(device)
            loss = loss + beta * (loss0 + (alpha * loss1) + (alpha * loss2) + (alpha * loss3))

            loss0.cpu().detach()
            loss1.cpu().detach()
            loss2.cpu().detach()
            loss3.cpu().detach()
            torch.cuda.empty_cache()

        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 100 == 0:
            #model.eval()
            visualize_so3(model, volume, label, output, output_so1, output_so2, output_so3, iteration, writer)
            #model.train()

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #        

def train_so4(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):
        # if volume_id == 0:
        #     hk = h5py.File(args.output+'/demo_label.h5','w')
        #     sz = (4, 320, 320)
        #     data = label[0].cpu().detach().squeeze().numpy().reshape(sz)
        #     hk.create_dataset('main', data=data, compression='gzip')
        #     hk.close()
        #     exit(0)
        # print('Iteration: ', iteration)
        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size

        # if i == 0: print(volume.size())
        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume = volume.squeeze().unsqueeze(1)
        if iteration==0: print(volume.size())

        criterion_so = WeightedBCELoss()
        # prepare label for side-outputs
        label_so0, label_so1, label_so2, label_so3, weight_so0, weight_so1, weight_so2, weight_so3 = prepare_so(label, device)

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output, output_so0, output_so1, output_so2, output_so3 = model(volume)
        if iteration==0: 
            print(output.size())
            print(class_weight.size())
            print(torch.max(class_weight), torch.min(class_weight))

        #print(torch.max(output))
        #print(torch.min(output))
        #assert (args.loss == 3 or args.loss == 4)
        lossk = criterion(output, label, class_weight)
        loss0 = criterion_so(output_so0, label_so0, weight_so0)
        loss1 = criterion_so(output_so1, label_so1, weight_so1)
        loss2 = criterion_so(output_so2, label_so2, weight_so2)
        loss3 = criterion_so(output_so3, label_so3, weight_so3)

        loss = lossk + 0.32 * loss0 + 0.16 * loss1 + 0.08 * loss2 + 0.04 * loss3
        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 100 == 0:
            #model.eval()
            visualize_so4(model, volume, label, output, 
                          output_so0, output_so1, output_so2, output_so3, 
                          iteration, writer, mask=class_weight)
            #model.train()

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #    

from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import square, disk
from skimage.morphology import skeletonize
from skimage.morphology import dilation

def corner_weight(img, k=0.06, sigma=0.4):
    coords = corner_peaks(corner_harris(img, k=k, sigma=sigma), min_distance=2)
    coords = np.array(coords)
    w_map = np.zeros(img.shape, dtype=np.uint8)
    w_map[coords[:,0], coords[:,1]] = 1
    w_map = dilation(w_map, disk(7))
    return w_map

def connect_aware_pred(output, class_weight, device, rule='union'):
    alpha = 4.0
    threshold = 0.3
    assert(rule in ['union', 'add'])

    prediction = output.clone() # (B, C, H, W) e.g. (16, 1, 512, 512)
    prediction = prediction.squeeze(1).detach().cpu().numpy()

    gt_weight = class_weight.clone()
    gt_weight = gt_weight.squeeze(1).detach().cpu().numpy()
    gt_weight = (gt_weight > 1.5).astype(np.uint8)

    weight = np.zeros(gt_weight.shape, dtype=np.uint8)
    for z in range(weight.shape[0]):
        img = prediction[z]
        img = img / np.max(img)
        img = (img > threshold).astype(np.uint8)
        img = skeletonize(img)
        weight[z] = corner_weight(img, k=0.06, sigma=0.4)

    # combine weight map from gt & pred
    if rule == 'add':
        final_weight =  gt_weight + weight
    elif rule == 'union':    
        final_weight =  np.logical_or(gt_weight, weight)
    final_weight = final_weight.astpye(np.float32) * alpha + 1.0
    final_weight = torch.from_numpy(final_weight.copy())
    final_weight = final_weight.unsqueeze(1) # (B, H, W) --> (B, C, H, W)
    return final_weight.to(device) 

# generate connectivity-aware weight
def train_ca(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for iteration, (volume, label, class_weight, _) in enumerate(train_loader):
        # if volume_id == 0:
        #     hk = h5py.File(args.output+'/demo_label.h5','w')
        #     sz = (4, 320, 320)
        #     data = label[0].cpu().detach().squeeze().numpy().reshape(sz)
        #     hk.create_dataset('main', data=data, compression='gzip')
        #     hk.close()
        #     exit(0)
        # print('Iteration: ', iteration)
        if iteration == 0:
            adjust_lr(args, optimizer, iteration, 0.01)
        if iteration == 100:
            adjust_lr(args, optimizer, iteration, 0.1)
        if iteration == 1000:
            adjust_lr(args, optimizer, iteration, 1.0)

        volume_id += args.batch_size

        # for gpu computing
        # print(weight_factor)
        volume = volume.squeeze().unsqueeze(1)
        if iteration==0: print(volume.size())

        criterion_so = WeightedBCELoss()
        # prepare label for side-outputs
        label_so0, label_so1, label_so2, label_so3, weight_so0, weight_so1, weight_so2, weight_so3 = prepare_so(label, device)

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output, output_so0, output_so1, output_so2, output_so3 = model(volume)
        if iteration==0: 
            print(output.size())
            print(class_weight.size())
            print(torch.max(class_weight), torch.min(class_weight))

        class_weight = connect_aware_pred(output_so0, class_weight, device)

        #print(torch.max(output))
        #print(torch.min(output))
        #assert (args.loss == 3 or args.loss == 4)
        lossk = criterion(output, label, class_weight)
        loss0 = criterion_so(output_so0, label_so0, weight_so0)
        loss1 = criterion_so(output_so1, label_so1, weight_so1)
        loss2 = criterion_so(output_so2, label_so2, weight_so2)
        loss3 = criterion_so(output_so3, label_so3, weight_so3)

        loss = lossk + 0.32 * loss0 + 0.16 * loss1 + 0.08 * loss2 + 0.04 * loss3
        writer.add_scalar('Loss', loss.item(), volume_id)        

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 100 == 0:
            #model.eval()
            visualize_so4(model, volume, label, output, 
                          output_so0, output_so1, output_so2, output_so3, 
                          iteration, writer, mask=class_weight)
            #model.train()

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        # Terminate
        if volume_id >= args.volume_total:
            break    #                
