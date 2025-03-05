import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
# gh:attention: in this script(networks), input_nc ≠ opt.input_nc. input_nc here represents the channels combine result.
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    else:
        raise 'generator has not been define!'
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=3, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class ConditionalConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConditionalConsistencyLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y, txt):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        txt_condition = torch.sigmoid(torch.mean(txt.detach()))
        conditon_loss = loss * txt_condition
        return conditon_loss


class ClassNLLoss(nn.Module):
    def __init__(self):
        super(ClassNLLoss, self).__init__()
        self.criterion = nn.NLLLoss()
    def forward(self, x, y):
        loss = 0
        for v in x:
            loss += self.criterion(v, y)
        return loss


class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)                

        for n in range(1, n_local_enhancers+1):
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]

            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        output_prev = self.model(input_downsampled[-1])        
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        e_model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            e_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                    norm_layer(ngf * mult * 2), activation]

        self.encoder = nn.Sequential(*e_model)

        r_model = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            r_model += [ResnetBlock(serial_num = i, dim = ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.r_blocks = nn.Sequential(*r_model)

        d_model = []         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            d_model += [nn.ConvTranspose2d(ngf * mult + 128 if i == 0 else ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        d_model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.decoder = nn.Sequential(*d_model)

        self.mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
 
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)


    def forward(self, input, txt):
        e = self.encoder(input)

        txt_data = txt[0]
        txt_len = txt[1]

        hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt_data.size(0)):
            mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt_data[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        h = (h_f + h_b) / 2
        cond = h.sum(0) / mask.sum(0)

        z_mean = self.mu(cond)
        z_log_stddev = self.log_sigma(cond)
        z = torch.randn(cond.size(0), 128, device=txt_data.device)
        cond = z_mean + z_log_stddev.exp() * z

        cond = cond.unsqueeze(-1).unsqueeze(-1)
        current_mean, current_std = cond.mean(), cond.std()
        new_mean, new_std = 0, 0.25
        cond = (cond - current_mean) / current_std * new_std + new_mean
        cond_ = cond.repeat(1, 1, e.size(2), e.size(3))
        merge = self.r_blocks(torch.cat((e, cond_), 1))

        temp_ = e + merge
        temp  = torch.concat((temp_, cond_), dim=1)
        d_result = self.decoder(temp)


        return d_result, (z_mean, z_log_stddev)
        

class ResnetBlock(nn.Module):
    def __init__(self, serial_num, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(serial_num, dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, serial_num, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        in_channels = dim + 128 if serial_num == 0 else dim
        conv_block += [nn.Conv2d(in_channels, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

    def txt_encoder(self, txt):
        txt_data = txt[0]
        txt_len = txt[1]

        hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt_data.size(0)):
            mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt_data[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        h = (h_f + h_b) / 2
        cond = h.sum(0) / mask.sum(0)

        z_mean = self.mu(cond)
        z_log_stddev = self.log_sigma(cond)
        z = torch.randn(cond.size(0), 128, device=txt_data.device)
        cond = z_mean + z_log_stddev.exp() * z
        
        return cond
    

    def mimi_acgan_D_result(self, input):
        _, _, v1, v2 = input.size()
        multi = v1*v2
        conv_layer = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=2).to(input.device)
        fc_realfake = nn.Linear(multi*512, 1).to(input.device)
        fc_classes = nn.Linear(multi*512, 2).to(input.device)
        sigmoid_ = nn.Sigmoid().to(input.device)
        softmax_ = nn.Softmax(dim=1).to(input.device)

        conv_out = conv_layer(input)

        input = input.view(-1, multi*512)
        realfake = sigmoid_(fc_realfake(input))
        realfake = realfake.view(-1, 1).squeeze(1)

        classes = softmax_(fc_classes(input))

        return conv_out, realfake, classes

    def singleD_forward(self, model, input, input_txt):
        if self.getIntermFeat:
            cat_input = torch.cat((input, input_txt), 1)
            result = [cat_input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            
            conv_out, realfake, classes = self.mimi_acgan_D_result(result[-1])
            result[-1] = conv_out
            return [result[1:], realfake, classes]
        else:
            cat_input = torch.cat((input, input_txt), 1)
            result = model(cat_input)
            conv_out, realfake, classes = self.mimi_acgan_D_result(result)
            result = conv_out
        return [result, realfake, classes]

    def forward(self, input, txt):
        txt = self.txt_encoder(txt) 
        txt = txt.unsqueeze(-1).unsqueeze(-1)
        txt = txt.repeat(1, 1, input.size(2), input.size(3))
        
        current_mean, current_std = txt.mean(), txt.std()
        new_mean, new_std = 0, 0.25
        txt = (txt - current_mean) / current_std * new_std + new_mean

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            
            result.append(self.singleD_forward(model, input_downsampled, txt))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
                txt = self.downsample(txt)
        result = [list(temp) for temp in zip(*result)]
        return result
        
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(0, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        
    
    """delete: actually, this part of code does not used by MultiDis"""
    # def forward(self, input):
    #     if self.getIntermFeat:
    #         res = [input]
    #         for n in range(self.n_layers+2):
    #             model = getattr(self, 'model'+str(n))
    #             res.append(model(res[-1]))
            
    #         # gh: modify and add two more output (realfake & classes).
    #         # gh: the first return is the original output of GridGAN
    #         realfake = [self.softmax_(output) for output in res[1:]]
    #         classes = [self.sigmoid_(output) for output in res[1:]]
    #         # return [res[1:], realfake, classes]
    #         return res[1:]
    #     else:
    #         output = self.model(input)
    #         realfake = self.softmax_(output)
    #         classes = self.sigmoid_(output)
    #         return [output, realfake, classes]

from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
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
