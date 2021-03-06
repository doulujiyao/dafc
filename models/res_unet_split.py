import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fsgan.utils.img_utils import create_pyramid


def make_conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, bias=False, padding_type='reflect',
                    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), use_dropout=False):
    """ Defines a convolution block with a normalization layer, an activation layer, and an optional dropout layer.

    Args:
        in_nc (int): Input number of channels
        out_nc (int): Output number of channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int, optional): The amount of padding for the height and width dimensions
        bias (bool): If True, enables the convolution's bias parameter
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5

    Returns:
        List of nn.Module. The list of the convolution block layers.
    """
    conv_block = []
    p = 0                           #就是在普通的卷积层之前可能加入padding层在之后加入 batchnorm relu dropout 
    if padding_type is not None:
        if padding_type == 'reflect':   #padding层 镜像填充
            conv_block += [nn.ReflectionPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(kernel_size // 2 if padding is None else padding)]
        elif padding_type == 'zero':
            p = kernel_size // 2 if padding is None else padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    elif padding is not None:
        p = padding

    conv_block.append(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=p, bias=bias))
    if norm_layer is not None:
        conv_block.append(norm_layer(out_nc))
    if act_layer is not None:
        conv_block.append(act_layer)

    if use_dropout:
        conv_block += [nn.Dropout(0.5)]

    return conv_block


class DownBlock(nn.Module):
    """ Defines a downsample block.

    Args:
        in_nc (int): Input number of channels
        out_nc (int): Output number of channels
        kernel_size (int): Convolution kernel size
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
    """
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(True)):
        super(DownBlock, self).__init__()   #利用stride 当作下采样
        model = make_conv_block(in_nc, out_nc, kernel_size, 2, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer)
        model += make_conv_block(out_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    """ Defines an upsample block.

    Args:
        in_nc (int): Input number of channels
        out_nc (int): Output number of channels
        kernel_size (int): Convolution kernel size
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
    """
    def __init__(self, in_nc, out_nc, kernel_size=3, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(True)):
        super(UpBlock, self).__init__()
        # model = [nn.Upsample(scale_factor=2, mode='bilinear')]
        model = make_conv_block(in_nc, out_nc, kernel_size, 1, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):   #利用插值当作上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.model(x)


class ResnetBlock(nn.Module):
    """ Defines a simple residual block.

    Args:
        planes: The input and output number of channels
        kernel_size (int): Convolution kernel size
        expansion (int): defines the intermediate number of channels (planes x expansion)
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
    """
    def __init__(self, planes, kernel_size=3, expansion=1, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        model = make_conv_block(planes, planes*expansion, kernel_size, padding_type=padding_type,
                                norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        model += make_conv_block(planes*expansion, planes, kernel_size, padding_type=padding_type,
                                 norm_layer=norm_layer, act_layer=None, use_dropout=False)
        self.model = nn.Sequential(*model)
        self.act = act_layer

    def forward(self, x):
        out = x + self.model(x)
        out = self.act(out)
        return out


class FlatBlock(nn.Module):
    """ Defines a block comprised of a number of residual blocks without that does not change the resolution
    or the number of channels.

    Args:
        planes: The input and output number of channels
        kernel_size (int): Convolution kernel size
        layers (int): The number of residual blocks
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
    """
    def __init__(self, planes, kernel_size=3, layers=1, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(True)):
        super(FlatBlock, self).__init__()
        if layers <= 0:
            self.model = None
        else:
            model = []
            for i in range(layers):   # 这个的layer 就是flat_layer的数值层数会扩张后收缩但是分辨率不变
                model.append(ResnetBlock(planes, kernel_size, 1, padding_type, norm_layer, act_layer))
            self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.model is None:
            return x
        return self.model(x)


# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|-- flat --|
class SkipConnectionBlock(nn.Module):
    """ Defines the skip connection block for the residual UNet.

    Args:
        ngf (int): Number of input and output channels
        sub_ngf (int): Number of input and output channels of the submodule
        down_block (nn.Module): Type of down block
        submodule (nn.Module): Submodule type
        up_block (nn.Module): Type of up block
        flat_block (nn.Module): Type of flat block
        flat_layers (int): Number of layers in the flat block
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
    """
    def __init__(self, ngf, sub_ngf, down_block=None, submodule=None, up_block=None, flat_block=None, flat_layers=1,
                 padding_type='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True), use_dropout=False):
        super(SkipConnectionBlock, self).__init__()
        self.submodule = submodule
        if submodule is not None:
            assert down_block is not None and up_block is not None
            self.down_block = down_block(ngf, sub_ngf, 3, padding_type, norm_layer, act_layer)
            self.up_block = up_block(sub_ngf, ngf, 3, padding_type, norm_layer, act_layer)
        if flat_block is not None:  #flat_block就是在up之后再通过flat_layers层的resnet后的结果
            self.flat_block = flat_block(ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        else:
            self.flat_block = None

    def forward(self, x):
        if self.submodule is not None:
            x = x + self.up_block(self.submodule(self.down_block(x)))
        if self.flat_block is not None:
            return self.flat_block(x)
        return x


class ResUNetatt(nn.Module):
    """ Defines the residual UNet architecture.

    Args:
        down_block (nn.Module): Type of down block
        up_block (nn.Module): Type of up block
        flat_block (nn.Module): Type of flat block
        in_nc (int): Input number of channels
        out_nc (tuple of ints): Output number of channels for each head
        max_nc (int): Maximum number of channels of the intermediate layers
        ngf (int): The number of input channels of the first intermediate layer
        flat_layers (list of ints): Number of layers in each flat block
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
    """
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=(3,), max_nc=None,
                 ngf=64, flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True),
                 use_dropout=False):
        super(ResUNetatt, self).__init__()   #ngf=128
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.in_conv = nn.Sequential(*make_conv_block(in_nc, ngf, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        max_nc = 1000000 if max_nc is None else max_nc
        self.out_conv=nn.Sequential(*make_conv_block(ngf, 1, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))

        # Output convolutions
        for i in range(len(out_nc)):
            out_conv = make_conv_block(ngf, out_nc[i], kernel_size=7, norm_layer=None, act_layer=None)
            out_conv.append(nn.Tanh())
            self.add_module('out_conv%d' % (i + 1),  nn.Sequential(*out_conv))

        self.levels = len(flat_layers)   #flat_layers=[0 2 3] 这个len就是几次下采样
        unet_block = None
        for i in range(1, self.levels + 1):   #i=1是底层 对应flat_layers的最后一层
            curr_ngf = min(ngf * (2 ** (self.levels - i)), max_nc)
            curr_sub_ngf = min(ngf * (2 ** (self.levels - i + 1)), max_nc)
            unet_block = SkipConnectionBlock(curr_ngf, curr_sub_ngf,
                                             down_block, unet_block, up_block, flat_block, flat_layers=flat_layers[-i],
                                             norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        self.inner = unet_block

    def forward(self, x):
        x = self.in_conv(x)
        x = self.inner(x)
        

        output = []
        for i in range(len(self.out_nc)):
            out_conv = getattr(self, 'out_conv%d' % (i + 1))
            output.append(out_conv(x))

        return tuple(output)


class ResUNet(nn.Module):
    """ Defines the residual UNet architecture.

    Args:
        down_block (nn.Module): Type of down block
        up_block (nn.Module): Type of up block
        flat_block (nn.Module): Type of flat block
        in_nc (int): Input number of channels
        out_nc (tuple of ints): Output number of channels for each head
        max_nc (int): Maximum number of channels of the intermediate layers
        ngf (int): The number of input channels of the first intermediate layer
        flat_layers (list of ints): Number of layers in each flat block
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
    """
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=(3,), max_nc=None,
                 ngf=64, flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True),
                 use_dropout=False):
        super(ResUNet, self).__init__()   #ngf=128
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.in_conv = nn.Sequential(*make_conv_block(in_nc, ngf, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        max_nc = 1000000 if max_nc is None else max_nc

        # Output convolutions
        for i in range(len(out_nc)):
            out_conv = make_conv_block(ngf, out_nc[i], kernel_size=7, norm_layer=None, act_layer=None)
            out_conv.append(nn.Tanh())
            self.add_module('out_conv%d' % (i + 1),  nn.Sequential(*out_conv))

        self.levels = len(flat_layers)   #flat_layers=[0 2 3] 这个len就是几次下采样
        unet_block = None
        for i in range(1, self.levels + 1):   #i=1是底层 对应flat_layers的最后一层
            curr_ngf = min(ngf * (2 ** (self.levels - i)), max_nc)
            curr_sub_ngf = min(ngf * (2 ** (self.levels - i + 1)), max_nc)
            unet_block = SkipConnectionBlock(curr_ngf, curr_sub_ngf,
                                             down_block, unet_block, up_block, flat_block, flat_layers=flat_layers[-i],
                                             norm_layer=norm_layer, act_layer=act_layer, use_dropout=use_dropout)
        self.inner = unet_block

    def forward(self, x):
        x = self.in_conv(x)
        x = self.inner(x)

        output = []
        for i in range(len(self.out_nc)):
            out_conv = getattr(self, 'out_conv%d' % (i + 1))
            output.append(out_conv(x))

        return tuple(output)


class LocalEnhancer(nn.Module):
    """ Define the architecture of the local enhancer described in:
    `"High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    <https://arxiv.org/pdf/1711.11585.pdf>`_.

    Args:
        ngf (int): Number of input and output channels
        sub_ngf (int): Number of input and output channels of the submodule
        down_block (nn.Module): Type of down block
        up_block (nn.Module): Type of up block
        flat_block (nn.Module): Type of flat block
        in_nc (int): Input number of channels
        out_nc (tuple of ints): Output number of channels for each head
        flat_layers (int): Number of layers in the flat block
        padding_type (str): The type of padding to perform ['reflect' | 'replicate' | 'zero']
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
    """
    def __init__(self, ngf, sub_ngf, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=(3,),
                 flat_layers=0, padding_type='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(inplace=True),
                 use_dropout=False):
        super(LocalEnhancer, self).__init__()
        self.down_block1 = down_block(ngf//2, sub_ngf//2, 3, padding_type, norm_layer, act_layer)
        self.down_block2 = down_block(ngf//2, sub_ngf//2, 3, padding_type, norm_layer, act_layer)
        if flat_block is not None:
            self.flat_block = flat_block(sub_ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        else:
            self.flat_block = None
        self.flat_block2= flat_block(sub_ngf, 3, flat_layers, padding_type, norm_layer, act_layer)
        self.up_block = up_block(sub_ngf, ngf, 3, padding_type, norm_layer, act_layer)
        self.in_conv1 = nn.Sequential(*make_conv_block(5, ngf//2, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))
        self.in_conv2 = nn.Sequential(*make_conv_block(5, ngf//2, kernel_size=7, norm_layer=norm_layer,
                                                      act_layer=act_layer, use_dropout=use_dropout))        
        
        # Output convolutions
        for i in range(len(out_nc)):
            out_conv = make_conv_block(ngf, out_nc[i], kernel_size=7, norm_layer=None, act_layer=None)
            out_conv.append(nn.Tanh())
            self.add_module('out_conv%d' % (i + 1), nn.Sequential(*out_conv))

    def extract_features(self, x, y,a1,a2):
        x1=x[:,0:4,:,:]
        x2=x[:,4:8,:,:]
        x2_1=x[:,8:11,:,:]
        l=x[:,11:,:,:]
        l1=x[:,3:4,:,:]
        l2=x[:,7:8,:,:]
        l1=torch.cat((l1,l),dim=1)
        l2=torch.cat((l2,l),dim=1)
        x1=torch.cat((x1,l),dim=1)
        x2=torch.cat((x2,l),dim=1)  
        x2_1=torch.cat((x2_1,l1),dim=1)
        x1=self.in_conv1(x1)
        x2=self.in_conv2(x2)
        x2_1=self.in_conv2(x2_1)
        x1=self.down_block1(x1)
        x2=self.down_block2(x2)
        x2_1=self.down_block2(x2_1)
        
        x_a=torch.cat((x2_1.mul(a1)+x2.mul(a2),x1),dim=1)
        x_a=self.flat_block2(x_a)

        x = x_a + y
        if self.flat_block is not None:
            x = self.flat_block(x)
        x = self.up_block(x)

        return x

    def forward(self, x, yx, y,a1,a2):

        x = self.extract_features(x, y,a1,a2)

        output = []
        for i in range(len(self.out_nc)):
            out_conv = getattr(self, 'out_conv%d' % (i + 1))
            output.append(out_conv(x))

        return tuple(output)


class MultiScaleResUNet(nn.Module):
    """ Defines a multi-scale variant of the residual UNet based on the progressive growing idea described in:
    `"High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    <https://arxiv.org/pdf/1711.11585.pdf>`_.

    Args:
        down_block (nn.Module): Type of down block
        up_block (nn.Module): Type of up block
        flat_block (nn.Module): Type of flat block
        in_nc (int): Input number of channels
        out_nc (tuple of ints): Output number of channels for each head
        max_nc (int): Maximum number of channels of the intermediate layers
        ngf (int): Number of input and output channels
        flat_layers (list of ints): Number of layers in each flat block
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        use_dropout (bool): If True, enables dropout with probability 0.5
        n_local_enhancers (int): Number of local enhancers
    """
    def __init__(self, down_block=DownBlock, up_block=UpBlock, flat_block=FlatBlock, in_nc=3, out_nc=(3,),
                 max_nc=None, ngf=64, flat_layers=(0, 0, 0, 3), norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(inplace=True), use_dropout=False, n_local_enhancers=1):
        super(MultiScaleResUNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_local_enhancers = n_local_enhancers

        # Global
        ngf_global = ngf * (2 ** n_local_enhancers)  #128 flat_layers=2 0 2 3
        self.base0 = ResUNet(down_block, up_block, flat_block, in_nc, out_nc, max_nc, ngf_global//2,
                            flat_layers[n_local_enhancers:-1], norm_layer, act_layer, use_dropout)
        self.base = ResUNet(down_block, up_block, flat_block, in_nc, out_nc, max_nc, ngf_global//2,
                            flat_layers[n_local_enhancers:-1], norm_layer, act_layer, use_dropout)
        self.base1 = ResUNetatt(down_block, up_block, flat_block, 2, out_nc, max_nc, 16,
                            flat_layers[n_local_enhancers:-1], norm_layer, act_layer, use_dropout)
        self.base2 = ResUNet(down_block, up_block, flat_block, ngf_global, out_nc, max_nc, ngf_global,
                            [0,1,2], norm_layer, act_layer, use_dropout)

        # Local enhancers
        for n in range(1, n_local_enhancers + 1):
            curr_ngf = ngf * (2**(n_local_enhancers-n)) #enhancer 对于高分辨率 卷积成多通道下分辨率和x相加后再一起升分辨率
            enhancer = LocalEnhancer(curr_ngf, curr_ngf * 2, down_block, up_block, flat_block, 12, out_nc,
                                     flat_layers[n - 1], 'reflect', norm_layer, act_layer, use_dropout)
            self.add_module('enhancer%d' % n, enhancer)

    def forward(self, pyd):
        pyd = create_pyramid(pyd, self.n_local_enhancers)
        x = pyd[-1]
        x1=x[:,0:4,:,:]
        x2=x[:,4:8,:,:]
        x2_1=x[:,8:11,:,:]
        l=x[:,11:,:,:]
        l1=x[:,3:4,:,:]
        l2=x[:,7:8,:,:]
        l1=torch.cat((l1,l),dim=1)
        l2=torch.cat((l2,l),dim=1)
        x1=torch.cat((x1,l),dim=1)
        x2=torch.cat((x2,l),dim=1)
        
        x2_1=torch.cat((x2_1,l1),dim=1)
        l1=self.base1.in_conv(l1)
        l1=self.base1.inner(l1)
        l1=self.base1.out_conv(l1)
        l2=self.base1.in_conv(l2)
        l2=self.base1.inner(l2)
        l2=self.base1.out_conv(l2)

        x1=self.base.in_conv(x1)
        x1=self.base.inner(x1)
        x2=self.base0.in_conv(x2)
        x2=self.base0.inner(x2)
        x2_1=self.base0.in_conv(x2_1)
        x2_1=self.base0.inner(x2_1)
        x_a=torch.cat((x2_1.mul(l1)+x2.mul(l2),x1),dim=1)




        # Call global at the coarsest level
        if len(pyd) == 1:

            return self.base2(x_a)  #用base的output层不要enhancers的output的层

        
        x = self.base2.in_conv(x_a)
        x = self.base2.inner(x)#不要base 的output 层

        # Apply enhancer for each level
        for n in range(1, len(pyd)):
            enhancer = getattr(self, 'enhancer%d' % n)
            # x = enhancer(pyd[self.n_local_enhancers - n], x)  之前的操作都是对于最后一层，也就是最小层，然后利用前面的层和x升分辨率
            #y=pyd[self.n_local_enhancers - n]
            #y_l=torch.cat((y[:,0:3,:,:]+x2.mul(l2),x1),dim=1)

            x = enhancer.extract_features(pyd[self.n_local_enhancers - n], x,a1=l1,a2=l2)
            if n == self.n_local_enhancers:
                output = []
                for i in range(len(self.out_nc)):  #enhancer后每一次 层数会小了2，因为下分率后层数乘以2才和x的层数相同
                    out_conv = getattr(enhancer, 'out_conv%d' % (i + 1))
                    output.append(out_conv(x))

                return tuple(output)

    def load_state_dict(self, state_dict, strict=True):
        # Find base in state_dict
        base_only_weights = True
        for name in state_dict.keys():
            if name.startswith('base'):
                base_only_weights = False

        if base_only_weights:
            self.base.load_state_dict(state_dict, strict)
        else:
            super(MultiScaleResUNet, self).load_state_dict(state_dict, strict)


def main(model='res_unet.ResUNet', res=(256,)):
    from fsgan.utils.obj_factory import obj_factory
    model = obj_factory(model)
    if len(res) == 1:
        img = torch.rand(1, model.in_nc, res, res)
        pred = model(img)
        print(pred.shape)
    else:
        img = []
        for i in range(1, len(res) + 1):
            img.append(torch.rand(1, model.in_nc, res[-i], res[-i]))
        pred = model(img)
        print(pred.shape)
        # for i in range(1, len(res) + 1):
        #     print(pred[-i].shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('res_unet test')
    parser.add_argument('model', default='res_unet.ResUNet',
                        help='model object')
    parser.add_argument('-r', '--res', default=(256,), type=int, nargs='+',
                        metavar='N', help='image resolution')
    args = parser.parse_args()
    main(args.model, args.res)
