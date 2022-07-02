import os
import re
import torch
import glob
import argparse
import skimage
import numpy as np
from skimage import io
from models import get_model
from utils.util import prctile_norm

opt = argparse.Namespace()

# opt.model = 'apcan_1_actin'
opt.model = 'apcan_3_actin'

# opt.model = 'apcan_1_er'
# opt.model = 'apcan_3_er'


opt.weights = 'pretrain/{}.pth'.format(opt.model)

# input/output layer options
opt.imageSize = 128
opt.scale = 2
opt.nch_in = 27
opt.nch_out = 1

# architecture options
opt.narch = 0
opt.n_resblocks = 4
opt.n_resgroups = 4
opt.reduction = 16
opt.n_feats = 64

# test options
opt.test = False
opt.cpu = False
opt.batchSize = 1
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')


def LoadModel(opt):
    print('Loading model')
    print(opt)
    net = get_model(opt)
    print('loading checkpoint', opt.weights)
    checkpoint = torch.load(opt.weights)
    if type(checkpoint) is dict:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    net.load_state_dict(state_dict)
    return net


def SIM_reconstruct27(model, opt):
    def prepimg(stack):
        input_27_frames = stack[:27]
        input_27_frames = input_27_frames.astype('float')
        input_27_frames = np.array(input_27_frames)
        for i in range(len(input_27_frames)):
            input_27_frames[i] = prctile_norm(input_27_frames[i])
        input_27_frames = torch.from_numpy(input_27_frames).float()
        return input_27_frames

    os.makedirs('%s' % opt.out, exist_ok=True)
    files = glob.glob('%s/*.tif' % opt.root)
    files = sorted(files, key=lambda name: int(re.findall(r"\d+\d*", name)[-1]))

    for iidx, imgfile in enumerate(files):
        print('[%d/%d] Reconstructing %s' % (iidx + 1, len(files), imgfile))
        stack = io.imread(imgfile)
        basename = os.path.basename(imgfile)
        print(stack.shape)
        print(basename[:-4])
        sim_raw_data = np.concatenate(
            (stack[0, :, :, :], stack[1, :, :, :], stack[2, :, :, :]))
        sim_input = prepimg(sim_raw_data)
        sim_input = sim_input.unsqueeze(0)
        with torch.no_grad():
            sr = model(sim_input.to(opt.device))
            sr = torch.clamp(sr.cpu(), min=0, max=1)
        sr = np.uint16(prctile_norm(sr.squeeze().numpy()) * 65535)
        skimage.io.imsave('%s/%s_%s.tif' % (opt.out, basename[:-4], opt.model), sr)


def SIM_reconstruct9(model, opt):
    def prepimg(stack):
        input_9_frames = stack[:9]
        input_9_frames = input_9_frames.astype('float')
        input_9_frames = np.array(input_9_frames)
        for i in range(len(input_9_frames)):
            input_9_frames[i] = prctile_norm(input_9_frames[i])
        input_9_frames = torch.from_numpy(input_9_frames).float()
        return input_9_frames

    os.makedirs('%s' % opt.out, exist_ok=True)
    files = glob.glob('%s/*.tif' % opt.root)
    files = sorted(files, key=lambda name: int(re.findall(r"\d+\d*", name)[-1]))

    for iidx, imgfile in enumerate(files):
        print('[%d/%d] Reconstructing %s' % (iidx + 1, len(files), imgfile))
        stack = io.imread(imgfile)
        basename = os.path.basename(imgfile)
        print(stack.shape)
        print(basename[:-4])
        sim_input = prepimg(stack[1, :, :, :])
        sim_input = sim_input.unsqueeze(0)
        with torch.no_grad():
            sr = model(sim_input.to(opt.device))
            sr = torch.clamp(sr.cpu(), min=0, max=1)
        sr = np.uint16(prctile_norm(sr.squeeze().numpy()) * 65535)
        skimage.io.imsave('%s/%s_%s.tif' % (opt.out, basename[:-4], opt.model), sr)


if __name__ == '__main__':
    net = LoadModel(opt)

    opt.root = './testing/actin'
    opt.out = './output'

    if opt.model[0:7] == "apcan_1":
        SIM_reconstruct9(net, opt)
    else:
        SIM_reconstruct27(net, opt)
