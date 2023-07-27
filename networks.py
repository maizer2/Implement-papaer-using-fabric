#coding=utf-8
import importlib, os, math, copy
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm

# from torchinfo import summary

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import init

from torchvision import models
import torch.nn.functional as F
import torch.optim as toptim
from torchvision.utils import make_grid

from visualization import save_images

# # # Custom
from diffusers import DiffusionPipeline, logging, DPMSolverMultistepScheduler


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def list_to_tuple(data):
    if isinstance(data, list):
        return tuple(data)
    elif isinstance(data, int):
        return (data, data)
    return data


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
 
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor
    
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
        
        
class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size
            P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)

            
    def forward(self, theta):
        grid_X, grid_Y = self.grid_X.cuda(), self.grid_Y.cuda()
        warped_grid = self.apply_transformation(theta,torch.cat((grid_X, grid_Y),3))
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        X, Y = X.cuda(), Y.cuda()
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N,1).fill_(1).cuda()
        Z = torch.FloatTensor(3,3).fill_(0).cuda()
        P = torch.cat((O,X,Y),1)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li
        
    def apply_transformation(self,theta,points):
        N = self.N
        P_X = self.P_X.cuda()
        P_Y = self.P_Y.cuda()
        P_X_base = self.P_X_base.cuda()
        P_Y_base = self.P_Y_base.cuda()
        
        Li = self.Li.cuda()
        
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords  
        
        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X=theta[:,:N,:,:].squeeze(3)
        Q_Y=theta[:,N:,:,:].squeeze(3)
        Q_X = Q_X + P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + P_Y_base.expand_as(Q_Y)
        
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = P_X.expand((1,points_h,points_w,1,self.N))
        P_Y = P_Y.expand((1,points_h,points_w,1,self.N))
        
        # compute weigths for non-linear part
        W_X = torch.bmm(Li[:,:N,:N].expand((batch_size,N,N)),Q_X)
        W_Y = torch.bmm(Li[:,:N,:N].expand((batch_size,N,N)),Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part
        A_X = torch.bmm(Li[:,N:,:N].expand((batch_size,3,N)),Q_X)
        A_Y = torch.bmm(Li[:,N:,:N].expand((batch_size,3,N)),Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3)
        points_Y_batch = points[:,:,:,1].unsqueeze(3)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
        
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        return torch.cat((points_X_prime,points_Y_prime),3)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)
    

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids
        
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DDPM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 1_000,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import DDPM library
        from pipeline.custom_DDPMPipeline import DDPMPipeline
        from diffusers import DDPMScheduler, UNet2DModel
        
        self.criterion = nn.MSELoss()
        
        self.sample_size = sample_size
        self.num_sampling = num_sampling
        
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        self.unet = UNet2DModel(sample_size=sample_size,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                down_block_types=("DownBlock2D",
                                                # "DownBlock2D",
                                                "DownBlock2D",
                                                "DownBlock2D",
                                                "AttnDownBlock2D",
                                                "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                # "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 128, 256, 256, 512),
                                layers_per_block=2)
        
        self.scheduler = DDPMScheduler(self.num_train_steps)
        self.pipeline = DDPMPipeline(self.unet, self.scheduler)
       
    
    # Sampling
    def reverse_latent_diffusion_process(self, xT = None) -> torch.FloatTensor:
        pred_x0 = self.pipeline(xT=xT,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_x0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            xT = self.forward_latent_diffusion_process(x0)
        else:
            xT = None
        
        pred_x0 = self.reverse_latent_diffusion_process(xT)
        
        return pred_x0
    
    
    def get_loss(self, x0):
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        xT = self.forward_latent_diffusion_process(x0, noise, t)
        eps_theta = self.unet(xT, t).sample
        
        loss = self.criterion(noise, eps_theta)
        return loss


class DDIM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import DDIM library
        from pipeline.custom_DDPMPipeline import DDPMPipeline
        from diffusers import DDIMScheduler, UNet2DModel
        
        self.criterion = nn.MSELoss()
        
        self.sample_size = list_to_tuple(sample_size)
        self.num_sampling = num_sampling
        
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        self.unet = UNet2DModel(sample_size=sample_size,
                                in_channels=in_channels,
                                out_channels=in_channels,
                                down_block_types=("DownBlock2D",
                                                 "DownBlock2D",
                                                 "DownBlock2D",
                                                 "DownBlock2D",
                                                 "AttnDownBlock2D",
                                                 "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 128, 256, 256, 512, 512),
                                layers_per_block=2)
        self.outnet = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
        self.scheduler = DDIMScheduler(self.num_train_steps)
        self.pipeline = DDPMPipeline(self.unet, self.scheduler)
       
    
    # Sampling
    def reverse_latent_diffusion_process(self, xT = None) -> torch.FloatTensor:
        pred_x0 = self.pipeline(xT=xT,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_x0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            xT = self.forward_latent_diffusion_process(x0)
        else:
            xT = None
        
        pred_x0 = self.reverse_latent_diffusion_process(xT)
        pred_x0 = self.outnet(pred_x0)
        
        return pred_x0
    
    
    def get_loss(self, x0, target = None):
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
        
        xT = self.forward_latent_diffusion_process(x0, noise, t)
        eps_theta = self.outnet(self.unet(xT, t).sample)
        
        
        if target is not None:
            loss = self.criterion(target, eps_theta)
        else:
            loss = self.criterion(noise, eps_theta)
        return loss
    
    
class UnconditionalLDM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import LDM library
        from pipeline.custom_LDMPipeline import UnconditionalLDMPipeline
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.models.vq_model import VQModel
        from diffusers.models.unet_2d import UNet2DModel
        
        self.criterion = nn.MSELoss()
        
        self.sample_size = list_to_tuple(sample_size)
        self.num_sampling = num_sampling
        
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        # self.vqvae = VQModel(in_channels=in_channels,
        #                      out_channels=out_channels,
        #                      down_block_types=("DownEncoderBlock2D",    #256 -> 128 -> 64 -> 32 / 192 -> 96 -> 48 -> 24
        #                                        "DownEncoderBlock2D",
        #                                        "DownEncoderBlock2D",
        #                                        "DownEncoderBlock2D"),
        #                      up_block_types=("UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",
        #                                      "UpDecoderBlock2D",),
        #                      block_out_channels=(64, 128, 256, 512),
        #                      layers_per_block=1,
        #                      latent_channels=out_channels,
        #                      sample_size=sample_size[0])
        self.vqvae = VQModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", ignore_mismatched_sizes=True)
        self.unet = UNet2DModel(sample_size=(32, 24),
                                in_channels=out_channels,
                                out_channels=out_channels,
                                down_block_types=("DownBlock2D",    # 32 -> 16 -> 8 -> 4 / 24 -> 12 -> 6 -> 3
                                                "AttnDownBlock2D",
                                                "DownBlock2D"),
                                up_block_types=("UpBlock2D",
                                                "AttnUpBlock2D",
                                                "UpBlock2D"),
                                block_out_channels=(128, 256, 512),
                                layers_per_block=2)
        
        self.scheduler = DDIMScheduler(self.num_train_steps)
        self.pipeline = UnconditionalLDMPipeline(vqvae=self.vqvae,
                                                 unet=self.unet,
                                                 scheduler=self.scheduler)
        
    
    def encode(self, x0):
        return self.vqvae.encode(x0).latents
    
    
    def decode(self, z0):
        return self.vqvae.decode(z0).sample
    
    
    # Sampling
    def reverse_latent_diffusion_process(self, latents = None) -> torch.FloatTensor:
        pred_z0 = self.pipeline(latents=latents,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_z0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        if x0 is not None:
            z = self.encode(x0)
            zT = self.forward_latent_diffusion_process(z)
        else:
            zT = None
        
        pred_z0 = self.reverse_latent_diffusion_process(zT)
        pred_x0 = self.decode(pred_z0)
        
        return pred_x0
    
    
    def get_loss(self, x0):
        z = self.encode(x0)
        t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        noise = torch.randn(z.shape, dtype=x0.dtype, device=x0.device)
        
        zT = self.forward_latent_diffusion_process(z, noise, t)
        print(zT.shape)
        eps_theta = self.unet(zT, t).sample
        print(eps_theta.shape)
        exit()
        loss = self.criterion(noise, eps_theta)
        return loss


class LDM(nn.Module):

    def __init__(self,
                 sample_size: Optional[Union[list, tuple, int]] = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_train_steps = 1_000,
                 num_inference_steps = 50,
                 num_sampling = 1,
                 eta: float = 0.0):
        super().__init__()
        ## import LDM library
        from diffusers import StableDiffusionImg2ImgPipeline
        
        self.criterion = nn.MSELoss()
        self.scheduler = None
        self.vae = None
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        
    
    def encode(self, x0):
        return self.vae.encode(x0).latents
    
    
    def decode(self, z0):
        return self.vae.decode(z0).sample
    
    
    # Sampling
    def reverse_latent_diffusion_process(self, latents = None) -> torch.FloatTensor:
        pred_z0 = self.pipeline(latents=latents,
                                num_sampling=self.num_sampling,
                                num_inference_steps=self.num_inference_steps)
        
        return pred_z0
    
        
    def forward_latent_diffusion_process(self, x0, noise = None, t = None) -> torch.FloatTensor:
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
            
        if t is None:
            t = torch.randint(0, self.num_train_steps, (x0.size(0), ), dtype=torch.long, device=x0.device)
        
        xT = self.scheduler.add_noise(x0, noise, t)
    
        return xT
    
    
    def forward(self, x0 = None):
        pass
    
    
    def get_loss(self, x0):
        pipe = self.pipeline.cuda()
        
        from torchvision import transforms
        
        x0 = transforms.Resize((256, 256))(x0)
        print(x0.shape)
        exit()
 

class GMM(nn.Module):
    """ Geometric Matching Module
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.criterionL1 = nn.L1Loss()
        
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d) 
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_dim=2*opt.grid_size**2, use_cuda=True)
        self.gridGen = TpsGridGen(opt.img_size[0], opt.img_size[1], use_cuda=True, grid_size=opt.grid_size)
        
    
    def get_image(self, batch, save_path): 
        warped_cloth, warped_mask = self(batch)
        
        warp_cloth_path = os.path.join(save_path, "warp-cloth")
        warp_mask_path = os.path.join(save_path, "warp-mask")
        
        os.makedirs(warp_cloth_path, exist_ok=True)
        os.makedirs(warp_mask_path, exist_ok=True)
        
        save_images(warped_cloth, batch["c_name"], warp_cloth_path)
        save_images(warped_mask, batch["c_name"], warp_mask_path)
        
        
    def forward(self, batch):
        agnostic = batch['agnostic'].cuda()
        c = batch['cloth'].cuda()
        cm = batch['cloth_mask'].cuda()
        
        featureA = self.extractionA(agnostic)
        featureB = self.extractionB(c)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        
        warped_cloth = F.grid_sample(c, grid, padding_mode='border',  align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros',  align_corners=True)
        
        return warped_cloth, warped_mask

    
    def get_loss(self, batch):
        im_c =  batch['parse_cloth'].cuda()
        warped_cloth, _ = self(batch)
        
        loss = self.criterionL1(warped_cloth, im_c)
        
        return loss
    
    
class TOM(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        
        self.opt = opt
        
        # self.model = DDPM(batch_size=opt.batch_size,
        #                   sample_size=opt.img_size,
        #                   in_channels=4,
        #                   out_channels=4,
        #                   num_train_steps=opt.num_train_steps,
        #                   num_inference_steps=opt.num_inference_steps,
        #                   eta=opt.eta)
        
        self.model = DDIM(sample_size=opt.img_size, 
                          in_channels=25,
                          out_channels=4,
                          num_train_steps=opt.num_train_steps,
                          num_inference_steps=opt.num_inference_steps,
                          num_sampling=3,
                          eta=opt.eta,)
        
        # self.model = LDM(batch_size=opt.batch_size,
        #                  sample_size=opt.img_size, 
        #                  in_channels=4, 
        #                  out_channels=4,
        #                  num_train_steps=opt.num_train_steps,
        #                  num_inference_steps=opt.num_inference_steps)
        
        
        # self.model = DiffAE(ffhq256_autoenc())
        
        # self.model = StableDiffusion_txt_to_img()
        
    
    def get_image(self, batch, save_path): 
        p_rendered, m_composite = self(batch)
        p_tryon = batch["cloth"] * m_composite + p_rendered * (1 - m_composite)
        
        p_rendered_path = os.apth.join(save_path, "p_rendered")
        m_composite_path = os.apth.join(save_path, "m_composite")
        tryon_path = os.apth.join(save_path, "tryon")
        
        os.makedirs(p_rendered_path, exist_ok=True)
        os.makedirs(m_composite_path, exist_ok=True)
        os.makedirs(tryon_path, exist_ok=True)
        
        save_images(p_rendered, batch["c_name"], p_rendered_path)
        save_images(m_composite, batch["c_name"], m_composite_path)
        save_images(p_tryon, batch["c_name"], tryon_path)
        
        
    def forward(self, batch):        
        agnostic, c = batch["agnostic"], batch["cloth"]
        x0 = torch.cat([agnostic, c], 1)
        outputs = self.model(x0)

        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered) # -1 ~ 1
        m_composite = F.sigmoid(m_composite) # 0 ~ 1
        
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)
        return p_rendered, m_composite, p_tryon
    

    def get_loss(self, batch):
        agnostic, c = batch["agnostic"], batch["cloth"]
        cm, image = batch['cloth_mask'], batch['image']
        
        x0 = torch.cat([agnostic, c], 1)
        target = torch.cat([image, cm], 1)
        
        return self.model.get_loss(x0, target)
        

class Lit_VTON(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        
        # configure
        self.opt = opt
        
        if opt.stage == "GMM":
            self.model = GMM(opt)
        elif opt.stage == "TOM":
            self.model = TOM(opt)
        
    
    def forward(self, batch):
        pass
        
    
    def sampling(self, batch):
        with torch.no_grad():
            if self.opt.stage == "GMM":
                warped_cloth, warped_mask = self.model(batch)
                
                self.logger.experiment.add_image("cloth", self.get_grid(batch["cloth"]), self.current_epoch)
                self.logger.experiment.add_image("cloth_mask", self.get_grid(batch["cloth_mask"]), self.current_epoch)
                self.logger.experiment.add_image("warped_cloth", self.get_grid(warped_cloth), self.current_epoch)
                self.logger.experiment.add_image("warped_mask", self.get_grid(warped_mask), self.current_epoch)

            elif self.opt.stage == "TOM":
                p_rendered, m_composite, tryon = self.model(batch)
                
                self.logger.experiment.add_image("Original", self.get_grid(batch["image"]), self.current_epoch)
                self.logger.experiment.add_image("Rendered", self.get_grid(p_rendered), self.current_epoch)
                self.logger.experiment.add_image("Cloth_mask", self.get_grid(batch["cloth_mask"]), self.current_epoch)
                self.logger.experiment.add_image("Composite", self.get_grid(m_composite), self.current_epoch)
                self.logger.experiment.add_image("Tryon", self.get_grid(tryon), self.current_epoch)
            
            
    def training_step(self, batch, batch_idx):
        
        loss = self.model.get_loss(batch)
        self.log("loss", loss, True, sync_dist=True)
        
        if self.trainer.is_last_batch:
            if self.current_epoch == 0:
                self.sampling(batch)
            elif (self.current_epoch + 1) % self.opt.sampling_step == 0:
                self.sampling(batch)
            
        return loss
    
    
    def configure_optimizers(self):
        optimizer = toptim.Adam(self.model.parameters(), self.opt.lr, betas=(0.5, 0.999))
        # lambda1 = lambda epoch: epoch // 30
        # lambda2 = lambda epoch: 0.95 ** epoch
        # lr_scheduler = toptim.lr_scheduler.LambdaLR(optimizer, [lambda1])
        
        return {"optimizer": optimizer,
                # "lr_scheduler": lr_scheduler
                }
    
    
    def get_grid(self, tensor, image_shape = None, normalize=True):
        if tensor.dtype is torch.bfloat16:
            tensor = tensor.type(torch.cuda.FloatTensor)
            
        if len(tensor.shape) == 2:
            tensor = tensor.view(tensor.size(0), image_shape[0], image_shape[1], image_shape[2])
        
        return make_grid(tensor, normalize=normalize)


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.model.get_image(batch, os.path.join(self.opt.dataroot, self.opt.datamode))
                
    