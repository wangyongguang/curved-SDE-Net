import math
import torch.nn as nn
import torch
import sys
import torch.nn.init as init
sys.path.append("C:\\Users\\milton\\PycharmProjects\\helloAI\\FGE+SDE-Net\\")
import curves
from scipy.special import binom
import numpy as np
__all__ = ['SDENet_cifar10Base','SDENet_cifar10Curve']


class Bezier(nn.Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                         padding=1, bias=False)
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class SDENet_cifar10Base(nn.Module):
    def __init__(self,layer_depth,num_classes=10,dim=64):
        super(SDENet_cifar10Base,self).__init__()
        self.layer_depth = layer_depth
        self.num_classes = num_classes
        self.dim = dim

        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim, dim, 4, 2, 1),
        )

        self.drift1 = nn.Sequential(
            #
            nn.ReLU(inplace=True),
            nn.Conv2d(dim+1,dim,3,1,1), #将ConcatConv2d换成Conv2d
            norm(dim),
        )
        self.drift2 = nn.Sequential(
            # norm(dim),
            nn.Conv2d(dim + 1, dim, 3, 1, 1),
            norm(dim),
        )

        self.diffusion1 = nn.Sequential(
            #
            nn.ReLU(inplace=True),
            nn.Conv2d(dim+1,dim,3,1,1),
            norm(dim),
        )
        self.diffusion2 = nn.Sequential(
            # norm(dim),
            nn.Conv2d(dim + 1, dim, 3, 1, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.fc_layers = nn.Sequential(
            norm(dim),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d((1, 1)),
                                       Flatten(),
                                       nn.Linear(dim, 10))
        self.deltat = 6. / self.layer_depth
        self.sigma = 50

    def forward(self,x,training_diffusion=False):
        out = self.downsampling_layers(x) #out.shape=(128,64,7,7)
        if not training_diffusion:
            t = 0
              #上面3行代替ConcatConv2d中的forward操作
            #ConcatConv2d中的操作本质上将x与tt在2维上进行合并，然后通过一个conv2d操作
            tt1 = torch.ones_like(out[:, :1, :, :]) * t
            ttx1 = torch.cat([tt1, out], 1) # ttx1.shape=(128,65,7,7)
            # print("out.shape={},ttx1.shape={}".format(out.shape,ttx1.shape))
            diffusion_term1 = self.sigma*self.diffusion1(ttx1)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1,2)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1,3)
            # print("diffusion_term1.shape={}".format(diffusion_term1.shape))
            tt2 = torch.ones_like(diffusion_term1[:, :1, :, :]) * t #有两个concatConv2d，
            ttx2 = torch.cat([tt2, diffusion_term1], 1)# 但是这个t不变，需要与中间值结合
            diffusion_term2 = self.diffusion2(ttx2)
            diffusion_term2 = torch.unsqueeze(diffusion_term2, 2)
            diffusion_term2 = torch.unsqueeze(diffusion_term2, 3)

            for i in range(self.layer_depth):
                t = 6 * (float(i)) / self.layer_depth
                tt1 = torch.ones_like(out[:, :1, :, :]) * t
                ttx1 = torch.cat([tt1, out], 1)

                drift_term = self.drift1(ttx1)
                tt2 = torch.ones_like(drift_term[:, :1, :, :]) * t
                ttx2 = torch.cat([tt2, drift_term], 1)
                out = out + self.drift2(ttx2)*self.deltat +diffusion_term2*\
                      math.sqrt(self.deltat)*torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
        else:
            t = 0
            tt1 = torch.ones_like(out[:, :1, :, :]) * t
            ttx1 = torch.cat([tt1, out], 1)
            diffusion_term1 = self.sigma * self.diffusion1(ttx1)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 2)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 3)

            tt2 = torch.ones_like(diffusion_term1[:, :1, :, :]) * t  # 有两个concatConv2d，
            ttx2 = torch.cat([tt2, diffusion_term1], 1)  # 但是这个t不变，需要与中间值结合
            final_out = self.diffusion2(ttx2)

        return final_out


class SDENet_cifar10Curve(nn.Module):
    def __init__(self, layer_depth, num_classes=10, dim=64,fix_points=[True,False,True]):
        super(SDENet_cifar10Curve, self).__init__()
        self.layer_depth = layer_depth
        self.num_classes = num_classes
        self.dim = dim
        self.deltat = 6./self.layer_depth
        self.sigma = 50

        self.coeff_layer = Bezier(3)

        self.downsampling_layers = nn.Sequential(
            curves.Conv2d(3, dim, 3, 1,fix_points=fix_points),
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
            nn.ReLU(inplace=True),
            curves.Conv2d(dim, dim, 4, 2, 1,fix_points=fix_points),
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
            nn.ReLU(inplace=True),
            curves.Conv2d(dim, dim, 4, 2, 1,fix_points=fix_points),
        )
        self.drift1 = nn.Sequential(
            nn.ReLU(inplace=True),
            curves.Conv2d(dim + 1, dim, 3, 1, 1,fix_points=fix_points),  # 将ConcatConv2d换成Conv2d
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
        )
        self.drift2 = nn.Sequential(
            # norm(dim),
            curves.Conv2d(dim + 1, dim, 3, 1, 1,fix_points=fix_points),
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
        )
        self.diffusion1 = nn.Sequential(
            # norm(dim),
            nn.ReLU(inplace=True),
            curves.Conv2d(dim + 1, dim, 3, 1, 1,fix_points=fix_points),
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
        )
        self.diffusion2 = nn.Sequential(
            # norm(dim),
            curves.Conv2d(dim + 1, dim, 3, 1, 1,fix_points=fix_points),
            curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            curves.Linear(dim, 1,fix_points=fix_points),
            nn.Sigmoid()
        )
        self.fc_layers = nn.Sequential(curves._GroupNorm(min(32, dim), dim,fix_points=fix_points),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d((1, 1)),
                                       Flatten(),
                                       curves.Linear(dim, num_classes,fix_points=fix_points))
        # Initialize weights
        for m in [self.downsampling_layers.modules(),self.drift1.modules(),
                  self.drift2.modules(),self.diffusion1.modules(),self.diffusion2.modules()]:
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()



    def forward(self, x,training_diffusion=False):
        t = x.data.new(1).uniform_()

        coeffs_t =  self.coeff_layer(t) #Bezier(3)

        out = self.downsampling_layers[0](x,coeffs_t)
        out = self.downsampling_layers[1](out,coeffs_t)
        out = self.downsampling_layers[2](out)
        out = self.downsampling_layers[3](out,coeffs_t)
        out = self.downsampling_layers[4](out,coeffs_t)
        out = self.downsampling_layers[5](out)
        out = self.downsampling_layers[6](out,coeffs_t)
        if not training_diffusion:
            t = 0
            # 上面3行代替ConcatConv2d中的forward操作
            # ConcatConv2d中的操作本质上将x与tt在2维上进行合并，然后通过一个conv2d操作
            tt1 = torch.ones_like(out[:, :1, :, :]) * t
            ttx1 = torch.cat([tt1, out], 1)

            diffusion1_term = self.diffusion1[0](ttx1) #nn.ReLU
            diffusion1_term = self.diffusion1[1](diffusion1_term,coeffs_t)
            diffusion1_term = self.diffusion1[2](diffusion1_term,coeffs_t)

            diffusion_term1 = self.sigma * diffusion1_term
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 2)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 3)

            tt2 = torch.ones_like(diffusion_term1[:, :1, :, :]) * t  # 有两个concatConv2d，
            ttx2 = torch.cat([tt2, diffusion_term1], 1)  # 但是这个t不变，需要与中间值结合

            diffusion2_term = self.diffusion2[0](ttx2,coeffs_t)
            diffusion2_term = self.diffusion2[1](diffusion2_term,coeffs_t)
            diffusion2_term = self.diffusion2[2](diffusion2_term)
            diffusion2_term = self.diffusion2[3](diffusion2_term)
            diffusion2_term = self.diffusion2[4](diffusion2_term)
            diffusion2_term = self.diffusion2[5](diffusion2_term,coeffs_t)
            diffusion_term2 = self.diffusion2[6](diffusion2_term)
            #
            # diffusion_term2 = self.diffusion2(ttx2,coeffs_t)
            diffusion_term2 = torch.unsqueeze(diffusion_term2, 2)
            diffusion_term2 = torch.unsqueeze(diffusion_term2, 3)

            for i in range(self.layer_depth):
                t = 6 * (float(i)) / self.layer_depth
                tt1 = torch.ones_like(out[:, :1, :, :]) * t
                ttx1 = torch.cat([tt1, out], 1)

                drift1_term = self.drift1[0](ttx1)
                drift1_term = self.drift1[1](drift1_term,coeffs_t)
                drift1_term = self.drift1[2](drift1_term,coeffs_t)
                # out = self.drift1(ttx1,coeffs_t)
                tt2 = torch.ones_like(drift1_term[:, :1, :, :]) * t
                ttx2 = torch.cat([tt2, drift1_term], 1)

                drift2_term = self.drift2[0](ttx2,coeffs_t)
                drift2_term = self.drift2[1](drift2_term,coeffs_t)

                out = out + drift2_term * self.deltat + diffusion_term2 * math.sqrt(
                    self.deltat) * torch.randn_like(out).to(x)

            final_out = self.fc_layers[0](out,coeffs_t)
            final_out = self.fc_layers[1](final_out)
            final_out = self.fc_layers[2](final_out)
            final_out = self.fc_layers[3](final_out)
            final_out = self.fc_layers[4](final_out,coeffs_t)
        else:
            t = 0
            tt1 = torch.ones_like(out[:, :1, :, :]) * t
            ttx1 = torch.cat([tt1, out], 1)

            diffusion1_term = self.diffusion1[0](ttx1)  # nn.ReLU
            diffusion1_term = self.diffusion1[1](diffusion1_term, coeffs_t)
            diffusion1_term = self.diffusion1[2](diffusion1_term, coeffs_t)

            diffusion_term1 = self.sigma * diffusion1_term

            # diffusion_term1 = self.sigma * self.diffusion1(ttx1,coeffs_t)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 2)
            # diffusion_term1 = torch.unsqueeze(diffusion_term1, 3)

            tt2 = torch.ones_like(diffusion_term1[:, :1, :, :]) * t  # 有两个concatConv2d，
            ttx2 = torch.cat([tt2, diffusion_term1], 1)  # 但是这个t不变，需要与中间值结合

            diffusion2_term = self.diffusion2[0](ttx2, coeffs_t)
            diffusion2_term = self.diffusion2[1](diffusion2_term, coeffs_t)
            diffusion2_term = self.diffusion2[2](diffusion2_term)
            diffusion2_term = self.diffusion2[3](diffusion2_term)
            diffusion2_term = self.diffusion2[4](diffusion2_term)
            diffusion2_term = self.diffusion2[5](diffusion2_term, coeffs_t)
            final_out = self.diffusion2[6](diffusion2_term)
            # final_out = self.diffusion2(ttx2,coeffs_t)
        return final_out


# class SDENet_cifar10:
#     base = SDENet_cifar10Base
#     curve = SDENet_cifar10Curve
#     kwargs = {'layer-depth':6}
#
# base = SDENet_cifar10Base(layer_depth=6, num_classes=10, dim=64)
# curve = SDENet_cifar10Curve(layer_depth=6, num_classes=10, dim=64,fix_points=[True,False,True])
# print(base)
# print(curve)
#
# print("base")
# print("====================================")
# for name,parameter in base.named_parameters():
#     print(name,':',parameter.size())
#
# print("curve")
# print("====================================")
# for name,parameter in curve.named_parameters():
#     print(name,':',parameter.size())
