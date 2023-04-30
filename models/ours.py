import torch
import torch.nn as nn
import torch.nn.functional as tnf

from models.registry import register_model
from models.entropic_student import InputBottleneck, BottleneckResNet, BottleneckResNetAuto, BottleneckResNetCondAuto

from compressai.layers.gdn import GDN1
from compressai.models.google import CompressionModel
import time


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class ResBlock(nn.Module):
    def __init__(self, in_out, hidden=None):
        super().__init__()
        hidden = hidden or (in_out // 2)
        self.conv_1 = nn.Conv2d(in_out, hidden, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(hidden, in_out, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.conv_1(tnf.gelu(input))
        x = self.conv_2(tnf.gelu(x))
        out = input + x
        return out


class ConvNeXtBlockAdaLN(nn.Module):
    def __init__(self, dim, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=2, residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        from timm.models.layers.mlp import Mlp
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x, emb):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x


class cconv(nn.Module):

    def __init__(self, in_channels, out_channels, embed_dim, kernel_size=3, stride=1):
        super(cconv, self).__init__()

        # Input shape (64, 52, 52)
        if in_channels < out_channels:
            group = in_channels
        else:
            group = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,groups=group)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        self.gelu1 = nn.GELU()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        # self.gelu2 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, out_channels)
        self.gamma = nn.Parameter(torch.full(size=(1, out_channels, 1, 1), fill_value=1e-6))
        self.fc3 = nn.Linear(embed_dim, out_channels)
        # self.softplus = nn.Softplus()

    def forward(self, x, emb):
        conv_out = self.conv(x)
        conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
        conv_out = self.norm(conv_out)
        conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        # conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
        # conv_out = self.norm(conv_out)
        # conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        # conv_out = self.gelu1(conv_out)
        # bpp_lmb = bpp_lmb.view(-1, 1)
        fc_out = self.gelu1(emb)
        fc_out = self.fc1(fc_out)
        fc_out1 = self.fc2(fc_out)
        fc_out1 = fc_out1.view(x.shape[0], -1, 1, 1)
        fc_out2 = self.fc3(fc_out)
        fc_out2 = fc_out2.reshape(x.shape[0], -1, 1, 1)

        fc_out2 = fc_out2 + 1
        # print("SHAPE ", conv_out.shape)
        # print("SHAPE ", fc_out1.shape)
        # print("SHAPE ", fc_out2.shape)
        conv_out = conv_out * fc_out2

        out = conv_out + fc_out1
        out = out.mul(self.gamma)

        return out

class cdeconv(nn.Module):

    def __init__(self, in_channels, out_channels, embed_dim, kernel_size=3, stride=1):
        super(cdeconv, self).__init__()

        # Input shape (64, 52, 52)
        if in_channels < out_channels:
            group = in_channels
        else:
            group = out_channels
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
        self.gelu1 = nn.GELU()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        # self.gelu2 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, out_channels)
        self.gamma = nn.Parameter(torch.full(size=(1, out_channels, 1, 1), fill_value=1e-6))
        self.fc3 = nn.Linear(embed_dim, out_channels)
        # self.softplus = nn.Softplus()

    def forward(self, x, emb):
        conv_out = self.deconv(x)
        conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
        conv_out = self.norm(conv_out)
        conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        # conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
        # conv_out = self.norm(conv_out)
        # conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
        # conv_out = self.gelu1(conv_out)
        # bpp_lmb = bpp_lmb.view(-1, 1)
        fc_out = self.gelu1(emb)
        fc_out = self.fc1(fc_out)
        fc_out1 = self.fc2(fc_out)
        fc_out1 = fc_out1.view(x.shape[0], -1, 1, 1)
        fc_out2 = self.fc3(fc_out)
        fc_out2 = fc_out2.reshape(x.shape[0], -1, 1, 1)

        fc_out2 = fc_out2 + 1
        # print("SHAPE ", conv_out.shape)
        # print("SHAPE ", fc_out1.shape)
        # print("SHAPE ", fc_out2.shape)
        conv_out = conv_out * fc_out2

        out = conv_out + fc_out1
        out = out.mul(self.gamma)

        return out

# class cconv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
#         super(cconv, self).__init__()

#         # Input shape (64, 52, 52)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
#         self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
#         self.gelu1 = nn.GELU()
#         self.embedding_layer = nn.Sequential(
#             nn.GELU(),
#             nn.Linear(1024, out_channels*2)
#         )

#     def forward(self, x, bpp_lmb):
#         conv_out = self.conv(x)

#         conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
#         conv_out = self.norm(conv_out)
#         conv_out = conv_out.permute(0, 3, 1, 2).contiguous()

#         B, C, H, W = conv_out.shape

#         # bpp_lmb = bpp_lmb.view(B, 1)
#         fc_out = self.embedding_layer(bpp_lmb)
#         assert fc_out.shape == (B, C*2)
#         fc_out = fc_out.view(B, C*2, 1, 1)
#         shift, scale = torch.chunk(fc_out, chunks=2, dim=1)

#         conv_out = conv_out * (1 + scale) + shift
#         # print(f"{conv_out.shape=}")
#         return conv_out

# class cdeconv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
#         super(cdeconv, self).__init__()

#         # Input shape (64, 52, 52)
#         self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)
#         self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
#         self.gelu1 = nn.GELU()
#         self.embedding_layer = nn.Sequential(
#             nn.GELU(),
#             nn.Linear(1024, out_channels*2)
#         )

#     def forward(self, x, bpp_lmb):
#         conv_out = self.conv(x)

#         conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
#         conv_out = self.norm(conv_out)
#         conv_out = conv_out.permute(0, 3, 1, 2).contiguous()

#         B, C, H, W = conv_out.shape

#         # bpp_lmb = bpp_lmb.view(B, 1)
#         fc_out = self.embedding_layer(bpp_lmb)
#         assert fc_out.shape == (B, C*2)
#         fc_out = fc_out.view(B, C*2, 1, 1)
#         shift, scale = torch.chunk(fc_out, chunks=2, dim=1)

#         conv_out = conv_out * (1 + scale) + shift
#         # print(f"{conv_out.shape=}")
#         return conv_out

# class cdeconv(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
#         super(cdeconv, self).__init__()

#         # Input shape (64, 52, 52)
#         if in_channels < out_channels:
#             group = in_channels
#         else:
#             group = out_channels
#         self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2, groups=group)
#         self.norm = nn.LayerNorm(out_channels, eps=1e-6, elementwise_affine=False)
#         self.gelu1 = nn.GELU()
#         self.fc1 = nn.Linear(1, 256)
#         # self.gelu2 = nn.GELU()
#         self.fc2 = nn.Linear(256, out_channels)
#         # self.fc3 = nn.Linear(256, out_channels)
#         # self.softplus = nn.Softplus()

#     def forward(self, x, bpp_lmb):
#         deconv_out = self.deconv(x)
#         deconv_out = deconv_out.permute(0, 2, 3, 1).contiguous()
#         deconv_out = self.norm(deconv_out)
#         deconv_out = deconv_out.permute(0, 3, 1, 2).contiguous()
#         # deconv_out = self.gelu1(deconv_out)
#         bpp_lmb = bpp_lmb.view(-1, 1)
#         fc_out = self.fc1(bpp_lmb)
#         fc_out = self.gelu1(fc_out)
#         fc_out1 = self.fc2(fc_out)
#         fc_out1 = fc_out1.view(x.shape[0], -1, 1, 1)
#         # fc_out2 = self.fc3(fc_out)
#         # fc_out2 = fc_out2.reshape(x.shape[0], -1, 1, 1)

#         fc_out2 = fc_out1 + 1
#         deconv_out = deconv_out * fc_out2

#         out = deconv_out + fc_out1

#         return out


class Bottleneck8(InputBottleneck):
    def __init__(self, hidden, zdim, num_target_channels=256, n_blocks=4):
        super().__init__(zdim)
        if n_blocks > 0:
            modules = [ResBlock(hidden) for _ in range(n_blocks)]
        else:
            modules = [nn.GELU()]
                                                                                    # Input = [3, 224, 224]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=8, stride=8, padding=0, bias=True),    # [128, 28, 28]
            *modules,                                                               # [128, 28, 28]
            nn.Conv2d(hidden, zdim, kernel_size=1, stride=1, padding=0),            # [64, 28, 28]
        )
        self.decoder = nn.Sequential(                                                                                           # Input = [64, 28, 28]
            deconv(zdim, num_target_channels),                                                                                  # [256, 56, 56]
            nn.GELU(),
            nn.Conv2d(num_target_channels, num_target_channels * 2, kernel_size=3, stride=1, padding=1, bias=True),             # [512, 56, 56]
            nn.GELU(),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=3, stride=1, padding=1, bias=True),             # [256, 56, 56]
            nn.GELU(),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=1, stride=1, padding=0, bias=True)                  # [256, 56, 56]
        )


class Autoencoder(InputBottleneck):
    def __init__(self, hidden, zdim, num_input_channels):
        super().__init__(zdim)

        self.encoder = nn.Sequential()                                                                                  # Input = [3, 224, 224] ; [256, 56, 56] ; [512, 28, 28] ; [1024, 14, 14] ;
        self.encoder.add_module("EN_Conv_1", conv(num_input_channels, hidden))                                          # [hidden, 112, 112] ; [hidden, 28, 28]
        self.encoder.add_module("EN_Gelu_1", nn.GELU())
        self.encoder.add_module("EN_Conv_2", conv(hidden, hidden*2, kernel_size=7, stride=1))                           # [hidden, 56, 56] ; [hidden*2, 28, 28]
        self.encoder.add_module("EN_Gelu_2", nn.GELU())
        self.encoder.add_module("EN_Conv_3", conv(hidden*2, hidden, kernel_size=7, stride=1))                           # [hidden, 28, 28] ; [hidden, 28, 28]
        self.encoder.add_module("EN_Gelu_3", nn.GELU())
        self.encoder.add_module("EN_Conv_4", conv(hidden, zdim, kernel_size=7, stride=1))                               # [zdim, 28, 28] ; [zdim, 28, 28]

        self.decoder = nn.Sequential()                                                                                  # Input = [zdim, 28, 28]
        self.decoder.add_module("DE_DeConv_1", deconv(zdim, hidden))                                                    # [hidden, 56, 56] ; [hidden, 56, 56]
        self.decoder.add_module("DE_Gelu_1", nn.GELU())
        self.decoder.add_module("DE_DeConv_2", deconv(hidden, hidden*2, kernel_size=7, stride=1))                       # [hidden*2, 112, 112] ; [hidden*2, 56, 56]
        self.decoder.add_module("DE_Gelu_2", nn.GELU())
        self.decoder.add_module("DE_DeConv_3", deconv(hidden*2, hidden, kernel_size=7, stride=1))                       # [hidden, 224, 224] ; [hidden, 56, 56]
        self.decoder.add_module("DE_Gelu_3", nn.GELU())
        self.decoder.add_module("DE_DeConv_4", deconv(hidden, num_input_channels, kernel_size=7, stride=1))             # [3, 224, 224] ; [256, 56, 56]


class CondEncoder(nn.Module):
    def __init__(self, hidden, zdim, num_input_channels, embed_dim):
        super().__init__()

        self.conv1 = conv(num_input_channels, hidden, kernel_size=5, stride=2)                                         # [hidden, 112, 112] ; [hidden, 28, 28]          # k=5, s=2 (downsampling)
        # self.gelu1 = nn.GELU()
        self.adaL1 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.conv2 = conv(hidden, hidden, kernel_size=3, stride=1)                          # [hidden, 56, 56] ; [hidden*2, 28, 28]
        # self.gelu2 = nn.GELU()
        self.adaL2 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.conv3 = conv(hidden, hidden, kernel_size=3, stride=1)                               # [zdim, 28, 28] ; [zdim, 28, 28]
        # self.gelu3 = nn.GELU()
        self.adaL3 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.adaL4 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        self.conv2 = conv(hidden, zdim, kernel_size=3, stride=1)

    def forward(self, x, emb):
        x = self.conv1(x)
        # x = self.gelu1(x)
        x = self.adaL1(x, emb)
        # x = self.conv2(x)
        x = tnf.gelu(x)
        x = self.adaL2(x, emb)
        # x = self.conv3(x)
        x = tnf.gelu(x)
        x = self.adaL3(x, emb)
        # x = tnf.gelu(x)
        # x = self.adaL4(x, emb)
        x = self.conv2(x)

    #     self.conv1 = cconv(num_input_channels, hidden, embed_dim)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu1 = nn.GELU()
    #     self.conv2 = cconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu2 = nn.GELU()
    #     self.conv3 = cconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu3 = nn.GELU()
    #     self.conv4 = cconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu4 = nn.GELU()
    #     self.conv5 = cconv(hidden, zdim, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]

    # def forward(self, x, emb):
    #     x = self.conv1(x, emb)
    #     x = self.gelu1(x)
    #     shortcut1 = x
    #     x = self.conv2(x, emb)
    #     x = self.gelu2(x)
    #     x += shortcut1
    #     shortcut2 = x
    #     x = self.conv3(x, emb)
    #     x = self.gelu3(x)
    #     x += shortcut2
    #     shortcut3 = x
    #     x = self.conv4(x, emb)
    #     x = self.gelu4(x)
    #     x += shortcut3
    #     x = self.conv5(x, emb)

        return x

class CondDecoder(nn.Module):
    def __init__(self, hidden, zdim, num_input_channels, embed_dim):
        super().__init__()

        self.deconv1 = deconv(zdim, hidden, kernel_size=5, stride=2) # [hidden, 112, 112] ; [hidden, 28, 28]
        # self.gelu1 = nn.GELU()
        self.adaL1 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.deconv2 = deconv(hidden, hidden, kernel_size=3, stride=1)                          # [hidden, 56, 56] ; [hidden*2, 28, 28]
        # self.gelu2 = nn.GELU()
        self.adaL2 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.deconv3 = deconv(hidden, hidden, kernel_size=3, stride=1)                               # [zdim, 28, 28] ; [zdim, 28, 28]
        # self.gelu3 = nn.GELU()
        self.adaL3 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        # self.adaL4 = ConvNeXtBlockAdaLN(hidden, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=4, residual=True, ls_init_value=1e-6)
        self.deconv2 = conv(hidden, num_input_channels, kernel_size=3, stride=1)

    def forward(self, x, emb):
        x = self.deconv1(x)
        # x = self.gelu1(x)
        x = self.adaL1(x, emb)
        # x = self.deconv2(x)
        x = tnf.gelu(x)
        x = self.adaL2(x, emb)
        # x = self.deconv3(x)
        x = tnf.gelu(x)
        x = self.adaL3(x, emb)
        # x = tnf.gelu(x)
        # x = self.adaL4(x, emb)
        x = self.deconv2(x)

    #     self.deconv1 = cdeconv(zdim, hidden, embed_dim)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu1 = nn.GELU()
    #     self.deconv2 = cdeconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu2 = nn.GELU()
    #     self.deconv3 = cdeconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu3 = nn.GELU()
    #     self.deconv4 = cdeconv(hidden, hidden, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]
    #     self.gelu4 = nn.GELU()
    #     self.deconv5 = cdeconv(hidden, num_input_channels, embed_dim, kernel_size=3, stride=1)                                         # [hidden, 112, 112] ; [hidden, 28, 28]

    # def forward(self, x, emb):
    #     x = self.deconv1(x, emb)
    #     x = self.gelu1(x)
    #     shortcut1 = x
    #     x = self.deconv2(x, emb)
    #     x = self.gelu2(x)
    #     x += shortcut1
    #     shortcut2 = x
    #     x = self.deconv3(x, emb)
    #     x = self.gelu3(x)
    #     x += shortcut2
    #     shortcut3 = x
    #     x = self.deconv4(x, emb)
    #     x = self.gelu4(x)
    #     x += shortcut3
    #     x = self.deconv5(x, emb)

        return x


class CondInputBottleneck(CompressionModel):
    def __init__(self, hidden, zdim, num_input_channels, embed_dim):
        super().__init__(entropy_bottleneck_channels=zdim)          # entropy_bottleneck_channels is the the number of channels of the latent representation(z) to be compressed
        self.encoder = CondEncoder(hidden, zdim, num_input_channels, embed_dim)
        self.decoder = CondDecoder(hidden, zdim, num_input_channels, embed_dim)
        self._flops_mode = False

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

    def encode(self, x, emb):
        z = self.encoder(x, emb)
        z_quantized, z_probs = self.entropy_bottleneck(z)
        return z_quantized, z_probs

    def forward(self, x, emb):
        z_quantized, z_probs = self.encode(x, emb)
        if self._flops_mode:
            return z_quantized, z_probs
        x_hat = self.decoder(z_quantized, emb)
        return x_hat, z_probs

    def update(self, force=False):
        return self.entropy_bottleneck.update(force=force)

    @torch.no_grad()
    def compress(self, x, emb):                                      # Compress latent representation (z) to char strings (z_strings)
        z = self.encoder(x, emb)
        start = time.perf_counter()
        compressed_z = self.entropy_bottleneck.compress(z)
        end = time.perf_counter()
        compression_time = (end - start) * 1000
        compressed_obj = (compressed_z, tuple(z.shape[2:]))
        return compressed_obj, compression_time

    @torch.no_grad()
    def decompress(self, compressed_obj, emb):                       # Decompress char strings (z_strings) to quantized latent representaion (z_quantized) and then decode it into reconstructed image x_hat
        bitstreams, latent_shape = compressed_obj
        z_quantized = self.entropy_bottleneck.decompress(bitstreams, latent_shape)
        feature = self.decoder(z_quantized, emb)
        return feature              # x_hat


# @register_model                                                                                     # ours_n8 = register_model(ours_n8)
# def ours_condautoencoder(num_classes=1000, bpp_lmb=None):
#     bottleneck = CondInputBottleneck(hidden=128, zdim=64, num_input_channels=512, embed_dim=1024)
#     model = BottleneckResNetCondAuto(zdim=64, embed_dim=1024, num_classes=num_classes, bpp_lmb=bpp_lmb, bottleneck_layer=bottleneck)
#     return model

@register_model                                                                                     # ours_n8 = register_model(ours_n8)
def ours_condautoencoder(num_classes=1000, bpp_lmb=None):
    in_channels = 256
    hidden = in_channels // 2
    zdim = in_channels // 8
    bottleneck = CondInputBottleneck(hidden=hidden, zdim=zdim, num_input_channels=in_channels, embed_dim=1024)
    model = BottleneckResNetCondAuto(zdim=zdim, embed_dim=1024, num_classes=num_classes, bpp_lmb=bpp_lmb, bottleneck_layer=bottleneck)
    return model

@register_model                                                                                     # ours_n8 = register_model(ours_n8)
def ours_autoencoder(num_classes=1000, bpp_lmb=1.28):
    bottleneck = Autoencoder(hidden=128, zdim=64, num_input_channels=256)
    model = BottleneckResNetAuto(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, bottleneck_layer=bottleneck)
    return model

@register_model                                                                                     # ours_n8 = register_model(ours_n8)
def ours_n8(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=8)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model                                                                                     # ours_n8_enc = register_model(ours_n8_enc)
def ours_n8_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=8)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def ours_n4(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck)
    return model

@register_model
def ours_n4_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=4)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model

@register_model
def ours_n0(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
         bottleneck_layer=bottleneck)
    return model

@register_model
def ours_n0_enc(num_classes=1000, bpp_lmb=1.28, teacher=True):
    bottleneck = Bottleneck8(hidden=128, zdim=64, num_target_channels=256, n_blocks=0)
    model = BottleneckResNet(zdim=64, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher,
                             bottleneck_layer=bottleneck, mode='encoder')
    return model
