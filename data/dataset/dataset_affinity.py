from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

from torch_connectomics.utils.seg.aff_util import seg_to_affgraph, affinitize
from torch_connectomics.utils.seg.seg_util import mknhood3d, widen_border1, widen_border2

from .dataset import BaseDataset
from .misc import crop_volume, rebalance_binary_class

class AffinityDataset(BaseDataset):
    """PyTorch ddataset class for affinity graph prediction.

    Args:
        volume: input image stacks.
        label: segmentation stacks.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor: data augmentor.
        mode (str): training or inference mode.
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 mode='train'):

        super(AffinityDataset, self).__init__(volume,
                                              label,
                                              sample_input_size,
                                              sample_label_size,
                                              sample_stride,
                                              augmentor,
                                              mode)

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        valid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            pos = self.get_pos_seed(vol_size, seed)
            out_label = crop_volume(self.label[pos[0]], vol_size, pos[1:])
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image':out_input, 'label':out_label}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'], augmented['label']
                out_input = out_input.astype(np.float32)
                out_label = out_label.astype(np.int16)
                #print(out_input.shape, out_label.shape) #debug
                #print(out_input.dtype, out_label.dtype) #debug

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])
            
        # Turn segmentation label into affinity in Pytorch Tensor
        if out_label is not None:
            # check for invalid region (-1)
            seg_bad = np.array([-1]).astype(out_label.dtype)[0]
            valid_mask = out_label!=seg_bad
            out_label[out_label==seg_bad] = 0
            out_label = widen_border1(out_label, 1)
            #out_label = widen_border2(out_label, 1)
            # replicate-pad the aff boundary
            out_label = seg_to_affgraph(out_label, mknhood3d(1), pad='replicate').astype(np.float32)
            out_label = torch.from_numpy(out_label.copy())

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input.copy())
        out_input = out_input.unsqueeze(0)

        if self.mode == 'train':
            # Rebalancing
            temp = 1.0 - out_label.clone()
            weight_factor, weight = rebalance_binary_class(temp)
            return pos, out_input, out_label, weight, weight_factor

        else:
            return pos, out_input

