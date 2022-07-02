import cv2
import torch
import torchvision
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image


def crop_image(img, d=32):
    imgsize = img.shape
    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)
    bbox = [
        int((imgsize[0] - new_size[0]) / 2),
        int((imgsize[1] - new_size[1]) / 2),
        int((imgsize[0] + new_size[0]) / 2),
        int((imgsize[1] + new_size[1]) / 2),
    ]
    img_cropped = img[0:new_size[0], 0:new_size[1], :]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def load(path):
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    img = load(path)
    if isinstance(imsize, int):
        imsize = (imsize, imsize)
    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)
    img_np = pil_to_np(img)
    return img, img_np


def fill_noise(x, noise_type):
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_init_sr(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [input_depth, 1, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var

    if method == 'zero':
        shape = [input_depth, 1, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
    return net_input


def get_input_kernel(path):
    img = io.imread(path)
    img_array = np.array(img)
    w, h = img_array.shape
    start = (w - 45) // 2
    img_array = img_array[start:start + 45, start:start + 45]
    img_tensor = torch.from_numpy(img_array).type(torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
    return img_tensor


def pil_to_np(img_PIL):
    ar = np.array(img_PIL)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32)


def np_to_pil(img_np):
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.
    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()
        print('Starting optimization with LBFGS')

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.1)  # learning rates
        for j in range(num_iter):
            scheduler.step(j)
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def readimg(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    return img, y, cb, cr
