import os
import re
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io
from utils.util import prctile_norm


class SIMDataset:
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)
        return parser

    def __init__(self, opt, category):
        super(SIMDataset, self).__init__()
        self.images_path = []
        if category == 'train':
            inputs = os.path.join(opt.root, 'training')
        elif category == 'valid':
            inputs = os.path.join(opt.root, 'validate')
        images_input_path = glob.glob(inputs + '/*')
        self.images_path.extend(images_input_path)
        self.scale = opt.scale
        self.task = opt.task
        self.nch_in = opt.nch_in
        self.nch_out = opt.nch_out
        self.data_norm = opt.data_norm
        self.out = opt.out
        self.model = opt.model
        self.category = category
        if category == 'valid':
            self.images_path = np.random.choice(self.images_path, size=opt.ntest)
        self.len = len(self.images_path)

    def __getitem__(self, index):
        img_path = glob.glob(self.images_path[index] + '/*.tif')
        img_path = sorted(img_path, key=lambda name: int(re.findall(r"\d+\d*", name)[-1]))
        stack = []
        for image_path in img_path:
            stack.append(io.imread(image_path))
        stack = np.array(stack).astype('float32')
        if self.category == 'train':
            gt = io.imread(self.images_path[index].replace('training', 'training_gt') + '.tif').astype('float32')
        elif self.category == 'valid':
            gt = io.imread(self.images_path[index].replace('validate', 'validate_gt') + '.tif').astype('float32')
        input_9_frames = stack[:self.nch_in]
        if self.model == 'srcnn':
            inputs = []
            w, h = input_9_frames[0].shape
            for i in range(len(input_9_frames)):
                inputs.append(
                    np.array(Image.fromarray(input_9_frames[i]).resize((h * 2, w * 2), resample=Image.BICUBIC)))
            wide_field = np.mean(inputs)
        else:
            wide_field = np.mean(input_9_frames, 0)

        # normalise
        if self.data_norm == 'minmax':
            gt = prctile_norm(gt)
            wide_field = prctile_norm(wide_field)
            for i in range(len(input_9_frames)):
                input_9_frames[i] = prctile_norm(input_9_frames[i])
        input_9_frames = torch.from_numpy(input_9_frames).float()
        wide_field = torch.from_numpy(wide_field).unsqueeze(0).float()
        gt = torch.from_numpy(gt).float()
        if self.nch_out == 1:
            gt = gt.unsqueeze(0)
        return {'sim_inputs': input_9_frames, 'sim_gt': gt, 'wf': wide_field}

    def __len__(self):
        return self.len
