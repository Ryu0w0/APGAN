import torch
from math import sqrt
from torch import nn
from torch.nn.utils import spectral_norm


class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, w, h = x.shape[0], x.shape[2], x.shape[3]
        std_feat_map = torch.std(x, dim=0)  # from (b, c, w, h) to (c, w, h)
        mean_std = torch.mean(std_feat_map)  # scalar
        std_ch = mean_std.expand((b, 1, w, h))  # expand scalar into (b, 1, w, h)
        x = torch.cat([std_ch, x], dim=1)  # concat toward channel
        return x


class PixelWiseNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        scale = torch.sqrt(torch.mean(x ** 2) + self.epsilon)
        x = x / scale
        return x


class EqualizedConv2d(nn.Module):
    """
    Normalize feature map by using the shape of weights.
    Although authors of PGAN proposed to normalize "weights", repeat to normalize weights make it smaller and smaller
    as process goes. Hence, feature maps are normalized instead of weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_equalize=True, use_norm=True, use_pixel_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.use_equalize = use_equalize
        self.w_scale = self.get_scale() if use_equalize else None
        self.use_norm = use_norm

        if not use_norm:
            self.norm = None
        elif use_pixel_norm:
            self.norm = PixelWiseNormalization()
        else:
            self.norm = nn.BatchNorm2d(in_channels)

    def get_scale(self):
        weight = getattr(self.conv, "weight")
        # weight.data.size(1): kernel_depth, weight.data[0][0].numel(): kernel_w * h
        f_in = weight.data.size(1) * weight.data[0][0].numel()
        scale = sqrt(2 / f_in)
        return scale

    def forward(self, x):
        x = self.conv(x)
        if self.use_equalize:
            x = x * self.w_scale
        if self.use_norm:
            x = self.norm(x)
        return x


class ConvBlockG(nn.Module):
    def __init__(self, block_in_ch, block_out_ch, config, for_1st_step=False, with_att=False):
        use_equalize = config["use_equalize"]
        use_pixel_norm = config["use_pixel_norm"]

        super().__init__()
        if for_1st_step:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=block_in_ch, out_channels=block_out_ch, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2),
                EqualizedConv2d(in_channels=block_out_ch, out_channels=block_out_ch,
                                use_equalize=use_equalize, use_norm=True, use_pixel_norm=use_pixel_norm),
                # PixelWiseNormalization(),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                EqualizedConv2d(in_channels=block_in_ch, out_channels=block_out_ch,
                                use_equalize=use_equalize, use_norm=True, use_pixel_norm=use_pixel_norm),
                # PixelWiseNormalization(),
                nn.LeakyReLU(negative_slope=0.2),
                EqualizedConv2d(in_channels=block_out_ch, out_channels=block_out_ch,
                                use_equalize=use_equalize, use_norm=True, use_pixel_norm=use_pixel_norm),
                # PixelWiseNormalization(),
                nn.LeakyReLU(negative_slope=0.2),
            )
            if with_att:
                self.main.add_module("attn", Self_Attn(block_out_ch))

    def forward(self, x):
        return self.main(x)


class ConvBlockD(nn.Module):
    def __init__(self, block_in_ch, block_out_ch, config, for_1st_step=False, with_std=True, with_att=False):
        super().__init__()
        use_equalize = config["use_equalize"]
        
        if for_1st_step:
            optional_module = [MiniBatchSTD()] if with_std else []
            fixed_module = [EqualizedConv2d(in_channels=block_in_ch+1, out_channels=block_out_ch,
                                            use_equalize=use_equalize, use_norm=False),
                            nn.LeakyReLU(negative_slope=0.2),
                            EqualizedConv2d(in_channels=block_out_ch, out_channels=block_out_ch, kernel_size=4, stride=1, padding=0,
                                            use_equalize=use_equalize, use_norm=False),
                            nn.LeakyReLU(negative_slope=0.2),
                            nn.Flatten(),
                            nn.Linear(in_features=512, out_features=1)]
            self.main = nn.Sequential(*(optional_module + fixed_module))
        else:
            self.main = nn.Sequential(
                EqualizedConv2d(in_channels=block_in_ch, out_channels=block_out_ch,
                                use_equalize=use_equalize, use_norm=False),
                nn.LeakyReLU(negative_slope=0.2),
                EqualizedConv2d(in_channels=block_out_ch, out_channels=block_out_ch,
                                use_equalize=use_equalize, use_norm=False),
                nn.LeakyReLU(negative_slope=0.2),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            )
            if with_att:
                self.main.add_module("attn", Self_Attn(block_out_ch))

    def forward(self, x):
        return self.main(x)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    """
    https://github.com/voletiv/self-attention-GAN-pytorch/blob/c1303d19341c2dcff47e361f0285a6c88e90ce9e/sagan_models.py
    """
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """
    Self attention Layer
    https://github.com/voletiv/self-attention-GAN-pytorch/blob/c1303d19341c2dcff47e361f0285a6c88e90ce9e/sagan_models.py
    """

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out