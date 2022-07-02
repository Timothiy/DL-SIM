import importlib
import torch.nn as nn
from torch.nn import init
from models.base_model import BaseModel
from models.APCAN_1 import APCAN as APCAN_1
from models.APCAN_3 import APCAN as APCAN_3


def find_model_using_name(model_name):
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)
    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model1 class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("checkpoint [%s] was created" % type(instance).__name__)
    return instance


def init_net(net, init_type='kaiming', init_gain=0.02, debug=False):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
    return net


def get_model(opt):
    print("---------------------------------{}-------------------------".format(opt.model))
    if opt.model.lower()[0:7] == 'apcan_1':
        net = APCAN_1(opt)
    elif opt.model.lower()[0:7] == 'apcan_3':
        net = APCAN_3(opt)
    else:
        print("model undefined")
        return None
    if not opt.cpu:
        net.cuda()
        net = nn.DataParallel(net)
    return init_net(net, init_type='normal', init_gain=0.02)
