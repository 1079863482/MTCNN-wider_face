import torchvision.transforms as transforms
import torch
from torch.autograd.variable import Variable
import numpy as np

transform = transforms.ToTensor()

def convert_image_to_tensor(image):
    """
    将数据转化为张量

    """

    return transform(image)


def convert_chwTensor_to_hwcNumpy(tensor):
    """
        把张量转化为数组并将轴换回来
            """

    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0,2,3,1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0,2,3,1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")