import itertools
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.layers.gdn import GDN1
from compressai.models.google import CompressionModel

from models.registry import register_model


class InputBottleneck(CompressionModel):
    def __init__(self, zdim):
        super().__init__(entropy_bottleneck_channels=zdim)          # entropy_bottleneck_channels is the the number of channels of the latent representation(z) to be compressed
        self.encoder: nn.Module
        self.decoder: nn.Module
        self._flops_mode = False

    def flops_mode_(self):
        self.decoder = None
        self._flops_mode = True

    def encode(self, x):
        z = self.encoder(x)
        z_quantized, z_probs = self.entropy_bottleneck(z)
        return z_quantized, z_probs

    def forward(self, x):
        z_quantized, z_probs = self.encode(x)
        if self._flops_mode:
            return z_quantized, z_probs
        x_hat = self.decoder(z_quantized)
        return x_hat, z_probs

    def update(self, force=False):
        return self.entropy_bottleneck.update(force=force)

    @torch.no_grad()
    def compress(self, x):                                      # Compress latent representation (z) to char strings (z_strings)
        z = self.encoder(x)
        compressed_z = self.entropy_bottleneck.compress(z)
        compressed_obj = (compressed_z, tuple(z.shape[2:]))
        return compressed_obj

    @torch.no_grad()
    def decompress(self, compressed_obj):                       # Decompress char strings (z_strings) to quantized latent representaion (z_quantized) and then decode it into reconstructed image x_hat
        bitstreams, latent_shape = compressed_obj
        z_quantized = self.entropy_bottleneck.decompress(bitstreams, latent_shape)
        feature = self.decoder(z_quantized)
        return feature              # x_hat


class BottleneckResNetLayerWithIGDN(InputBottleneck):
    def __init__(self, num_enc_channels=24, num_target_channels=256):
        super().__init__(num_enc_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )


class BottleneckResNet(nn.Module):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, teacher=True, mode='joint',
                 bottleneck_layer=None):
        super().__init__()
        if bottleneck_layer is None:
            bottleneck_layer = BottleneckResNetLayerWithIGDN(zdim, 256)
        self.bottleneck_layer = bottleneck_layer

        from torchvision.models.resnet import resnet50, ResNet50_Weights
        resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
        # if mode == 'encoder':
        #     for p in resnet_model.parameters():
        #         p.requires_grad_(False)
        #     for m in resnet_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.track_running_stats = False
        #         debug = 1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc

        if teacher:
            from models.teachers import ResNetTeacher
            self._teacher = ResNetTeacher(source='torchvision')
            for p in self._teacher.parameters():
                p.requires_grad_(False)
        else:
            self._teacher = None

        self.bpp_lmb = float(bpp_lmb)

        self.train_mode = mode
        if mode == 'encoder':
            for p in itertools.chain(self.layer2.parameters(), self.layer3.parameters(),
                                     self.layer4.parameters(), self.fc.parameters()):
                p.requires_grad_(False)
            self.lambdas = [1.0, 1.0, self.bpp_lmb] # cls, trs, bpp
        elif mode == 'classifier':
            raise DeprecationWarning()
            for p in self.bottleneck_layer.encoder.parameters():
                p.requires_grad_(False)
            for p in self.bottleneck_layer.entropy_bottleneck.parameters():
                p.requires_grad_(False)
            self.lambdas = [1.0, 0.0, 0.0] # cls, trs, bpp
        elif mode == 'joint':
            self.lambdas = [1.0, 1.0, self.bpp_lmb] # cls, trs, bpp
        else:
            raise ValueError()

        self.compress_mode = False

    def train(self, mode=True):
        super().train(mode)
        if self.train_mode == 'encoder': # make classifier and teacher always eval
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.fc.eval()
            if self._teacher is not None:
                self._teacher.eval()
        return self

    def compress_mode_(self):
        self.bottleneck_layer.update(force=True)
        self.compress_mode = True

    def forward(self, x, y):
        nB, _, imH, imW = x.shape
        x1, p_z = self.bottleneck_layer(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)

        probs_teach, features_teach = self.forward_teacher(x)

        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # label prediction loss
        l_ce = tnf.cross_entropy(logits_hat, y, reduction='mean')

        l_kd = tnf.kl_div(input=torch.log_softmax(logits_hat, dim=1),           # why one is using softmax and the other log_softmax???
                          target=probs_teach, reduction='batchmean')
        # transfer loss
        lmb_cls, lmb_trs, lmb_bpp = self.lambdas
        if lmb_trs > 0:
            l_trs = self.transfer_loss([x1, x2, x3, x4], features_teach)
        else:
            l_trs = [torch.zeros(1, device=x.device) for _ in range(4)]
        loss = lmb_cls * (0.5*l_ce + 0.5*l_kd) + lmb_trs * sum(l_trs) + lmb_bpp * bppix

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.item()
        stats['CE'] = l_ce.item()
        stats['KD'] = l_kd.item()
        for i, lt in enumerate(l_trs):
            stats[f'trs_{i}'] = lt.item()
        stats['acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)              # shouldn't logits_hat be passed through a softmax first????
        if lmb_trs > 0:
            stats['t_acc'] = (torch.argmax(probs_teach, dim=1) == y).sum().item() / float(nB)
        else:
            stats['t_acc'] = -1.0
        return stats

    @torch.no_grad()
    def forward_teacher(self, x):
        y_teach = self._teacher(x)
        t1, t2, t3, t4 = self._teacher.cache
        assert all([(not t.requires_grad) for t in (t1, t2, t3, t4)])
        assert y_teach.dim() == 2
        y_teach = torch.softmax(y_teach, dim=1)
        return y_teach, (t1, t2, t3, t4)

    def transfer_loss(self, student_features, teacher_features):
        losses = []
        for fake, real in zip(student_features, teacher_features):
            if (fake is not None) and (real is not None):
                assert fake.shape == real.shape, f'fake{fake.shape}, real{real.shape}'
                l_trs = tnf.mse_loss(fake, real, reduction='mean')
                losses.append(l_trs)
            else:
                device = next(self.parameters()).device
                losses.append(torch.zeros(1, device=device))
        return losses

    @torch.no_grad()
    def self_evaluate(self, x, y):
        # raise NotImplementedError()
        nB, _, imH, imW = x.shape
        if self.compress_mode:
            compressed_obj = self.bottleneck_layer.compress(x)
            num_bits = get_object_size(compressed_obj)
            x1 = self.bottleneck_layer.decompress(compressed_obj)
        else:
            x1, p_z = self.bottleneck_layer(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')

        _, top5_idx = torch.topk(logits_hat, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(y.cpu().view(nB, 1), top5_idx.cpu())
        stats = OrderedDict()
        stats['top1'] = correct5[:, 0].float().mean().item()
        stats['top5'] = correct5.any(dim=1).float().mean().item()
        if self.compress_mode:
            bppix = num_bits / float(nB * imH * imW)
        else:
            bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix
        stats['loss'] = float(l_cls + self.bpp_lmb * bppix)
        return stats

    def state_dict(self):
        msd = super().state_dict()
        for k in list(msd.keys()):
            if '_teacher' in k:
                msd.pop(k)
        return msd

    def update(self):
        self.bottleneck_layer.update()

    @torch.no_grad()
    def send(self, x):
        compressed_obj = self.bottleneck_layer.compress(x)
        return compressed_obj

    @torch.no_grad()
    def receive(self, compressed_obj):
        feature = self.bottleneck_layer.decompress(compressed_obj)
        x2 = self.layer2(feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        p_logits = self.fc(feature)
        return p_logits


class BottleneckResNetAuto(nn.Module):
    def __init__(self, zdim=24, num_classes=1000, bpp_lmb=0.02, bottleneck_layer=None):
        super().__init__()

        self.bottleneck_layer = bottleneck_layer

        from torchvision.models.resnet import resnet50, ResNet50_Weights
        resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model.eval()

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.act1  = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc

        self.bpp_lmb = float(bpp_lmb)
        self.compress_mode = False

    def compress_mode_(self):
        self.bottleneck_layer.update(force=True)
        self.compress_mode = True

    def forward(self, x, y):
        nB, _, imH, imW = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x1_hat, p_z = self.bottleneck_layer(x1)
        x2 = self.layer2(x1_hat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)

        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # label prediction loss
        l_ce = tnf.cross_entropy(logits_hat, y, reduction='mean')
        loss = l_ce + self.bpp_lmb * bppix
        # aux_loss = self.bottleneck_layer.aux_loss()

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.item()
        stats['CE'] = l_ce.item()
        stats['acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)
        # stats['aux_loss'] = aux_loss

        return stats

    @torch.no_grad()
    def self_evaluate(self, x, y):
        # raise NotImplementedError()
        nB, _, imH, imW = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        if self.compress_mode:
            compressed_obj = self.bottleneck_layer.compress(x)
            num_bits = get_object_size(compressed_obj)
            x1 = self.bottleneck_layer.decompress(compressed_obj)
        else:
            x1 = self.layer1(x)
            x1_hat, p_z = self.bottleneck_layer(x1)
        x2 = self.layer2(x1_hat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')

        _, top5_idx = torch.topk(logits_hat, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(y.cpu().view(nB, 1), top5_idx.cpu())
        stats = OrderedDict()
        stats['top1'] = correct5[:, 0].float().mean().item()
        stats['top5'] = correct5.any(dim=1).float().mean().item()
        if self.compress_mode:
            bppix = num_bits / float(nB * imH * imW)
        else:
            bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        stats['bppix'] = bppix
        stats['loss'] = float(l_cls + self.bpp_lmb * bppix)
        return stats

    def update(self):
        self.bottleneck_layer.update()

    @torch.no_grad()
    def send(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        compressed_obj = self.bottleneck_layer.compress(x)
        return compressed_obj

    @torch.no_grad()
    def receive(self, compressed_obj):
        feature = self.bottleneck_layer.decompress(compressed_obj)
        x2 = self.layer2(feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        p_logits = self.fc(feature)
        return p_logits


class BottleneckResNetCondAuto(nn.Module):
    def __init__(self, zdim=24, embed_dim=1024, num_classes=1000, bpp_lmb=None, bottleneck_layer=None):
        super().__init__()

        self.bottleneck_layer = bottleneck_layer

        from torchvision.models.resnet import resnet50, ResNet50_Weights
        resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model.eval()

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.act1  = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.fc = resnet_model.fc

        self.embedding_block = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

        # self.eval_mode = False
        if bpp_lmb is not None:
            self.bpp_lmb = bpp_lmb
            # self.eval_mode = True
        self.compress_mode = False

    def compress_mode_(self):
        self.bottleneck_layer.update(force=True)
        self.compress_mode = True

    def forward(self, x, y):
        nB, _, imH, imW = x.shape
        device = x.device
        # if not self.eval_mode:
        # self.bpp_lmb = torch.rand(nB)*5.12 + 0.08

        ########## normal sampling ##############
        # lmb_min, lmb_max = -5.12, 10.24
        # self.bpp_lmb = lmb_min + (lmb_max - lmb_min) * torch.rand(nB, 1, device=device)

        ########## log sampling ##############
        log_lmb_min, log_lmb_max = math.log(0.0000001), math.log(10.24)      # math.log(0.0000001), math.log(10.24)
        # distribution = torch.distributions.uniform.Uniform(low=log_lmb_min, high=log_lmb_max)
        # self.bpp_lmb = torch.exp(distribution.sample(sample_shape=torch.Size([nB, 1])))
        self.bpp_lmb = torch.exp(log_lmb_min + (log_lmb_max - log_lmb_min) * torch.rand(nB, 1, device=device))

        # self.bpp_lmb = torch.rand(nB)
        # self.bpp_lmb = self.bpp_lmb.view(-1, 1)
        # self.bpp_lmb = torch.tensor([1.28]).expand(nB).view(-1, 1)
        emb = self.embedding_block(self.bpp_lmb)
        # else:
        #     self.bpp_lmb = self.bpp_lmb.expand(nB)
        # self.bpp_lmb = self.bpp_lmb.to(device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x2 = self.layer2[:2](x1)
        # x3 = self.layer3[:3](x2)
        # x3 = self.layer3(x2)
        x1_hat, p_z = self.bottleneck_layer(x1, emb)           # p_z has shape [batch_size, zdim, z_w, z_h]
        # x2 = self.layer2[2:4](x2_hat)
        x2 = self.layer2(x1_hat)
        # x3 = self.layer3[3:6](x3_hat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)

        # bppix = -1.0 * torch.log2(p_z).mean(0).sum() / float(imH * imW)
        # bit rate loss
        bppix = -1.0 * torch.log2(p_z).sum(dim=(1, 2, 3)) / float(imH * imW)
        # label prediction loss
        l_ce = tnf.cross_entropy(logits_hat, y, reduction='mean')
        mse = tnf.mse_loss(x1, x1_hat)
        # if not self.eval_mode:
        # self.bpp_lmb = self.bpp_lmp.item()

        # loss = l_ce + (self.bpp_lmb.squeeze() * bppix).mean()
        loss = 256 * mse + (self.bpp_lmb.squeeze() * bppix).mean()

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bppix'] = bppix.mean().item()
        stats['CE'] = l_ce.item()
        stats['acc'] = (torch.argmax(logits_hat, dim=1) == y).sum().item() / float(nB)
        stats['mse'] = mse.item()
        stats['PSNR'] = -10 * math.log10(mse)
        # stats['aux_loss'] = aux_loss

        return stats

    @torch.no_grad()
    def self_evaluate(self, x, y):
        # raise NotImplementedError()
        nB, _, imH, imW = x.shape
        device = x.device
        lmb_min = torch.tensor([0.0000001]).expand(nB).view(-1, 1).to(device)        # 0.08
        lmb_max = torch.tensor([10.24]).expand(nB).view(-1, 1).to(device)        # 5.12
        # print(f"Evaluating for lambda = {lmb_min.item()}")
        nB, _, imH, imW = x.shape
        emb = self.embedding_block(lmb_min)
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.act1(x0)
        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        # x2 = self.layer2(x1)
        # x2 = self.layer2[:2](x1)
        # x3 = self.layer3[:3](x2)
        # x3 = self.layer3(x2)
        x1_hat, p_z = self.bottleneck_layer(x1, emb)           # p_z has shape [batch_size, zdim, z_w, z_h]
        # x2 = self.layer2[2:4](x2_hat)
        # x3 = self.layer3[3:6](x3_hat)
        x2 = self.layer2(x1_hat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')
        mse = tnf.mse_loss(x1, x1_hat)

        _, top5_idx = torch.topk(logits_hat, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(y.cpu().view(nB, 1), top5_idx.cpu())
        stats = OrderedDict()
        stats['top1_min'] = correct5[:, 0].float().mean().item()
        # stats['top5_min'] = correct5.any(dim=1).float().mean().item()
        bppix = -1.0 * torch.log2(p_z).sum(dim=(1, 2, 3)) / float(imH * imW)
        stats['bppix_min'] = bppix.mean(0).item()
        stats['loss_min'] = float(l_cls + (lmb_min.squeeze() * bppix).mean(0))
        stats['PSNR_min'] = -10 * math.log10(mse)

        # print(f"Evaluating for lambda = {lmb_max.item()}")
        nB, _, imH, imW = x.shape
        emb = self.embedding_block(lmb_max)
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.act1(x0)
        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        # x2 = self.layer2(x1)
        # x2 = self.layer2[:2](x1)
        # x3 = self.layer3[:3](x2)
        # x3 = self.layer3(x2)
        x1_hat, p_z = self.bottleneck_layer(x1, emb)           # p_z has shape [batch_size, zdim, z_w, z_h]
        # x2 = self.layer2[2:4](x2_hat)
        # x3 = self.layer3[3:6](x3_hat)
        x2 = self.layer2(x1_hat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        logits_hat = self.fc(feature)
        l_cls = tnf.cross_entropy(logits_hat, y, reduction='mean')
        mse = tnf.mse_loss(x1, x1_hat)

        _, top5_idx = torch.topk(logits_hat, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(y.cpu().view(nB, 1), top5_idx.cpu())
        stats['top1_max'] = correct5[:, 0].float().mean().item()
        # stats['top5_max'] = correct5.any(dim=1).float().mean().item()
        bppix = -1.0 * torch.log2(p_z).sum(dim=(1, 2, 3)) / float(imH * imW)
        stats['bppix_max'] = bppix.mean(0).item()
        stats['loss_max'] = float(l_cls + (lmb_max.squeeze() * bppix).mean(0))
        stats['PSNR_max'] = -10 * math.log10(mse)

        return stats

    def update(self):
        self.bottleneck_layer.update()

    @torch.no_grad()
    def send(self, x):
        # self.bpp_lmb = self.bpp_lmb.expand(x.shape[0])
        import time
        time_list = [0, 0, 0, 0]
        device = x.device
        self.bpp_lmb_vec = torch.tensor([self.bpp_lmb]).expand(x.shape[0]).view(-1, 1).to(device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        start_time = time.perf_counter()
        start = time.perf_counter()
        x = self.layer1(x)
        # x = self.layer2[:2](x)
        # x = self.layer2[2:4](x)
        end = time.perf_counter()
        time_list[0] = (end - start) * 1000
        # start = time.perf_counter()

        # end = time.perf_counter()
        # time_list[1] = (end - start) * 1000
        # start = time.perf_counter()

        # end = time.perf_counter()
        # time_list[2] = (end - start) * 1000
        # x = self.layer2(x)
        # x = self.layer3[:3](x)
        # x = self.layer3(x)
        # start = time.perf_counter()
        emb = self.embedding_block(self.bpp_lmb_vec)
        # end = time.perf_counter()
        # time_list[0] = (end - start) * 1000
        start = time.perf_counter()
        compressed_obj, compression_time = self.bottleneck_layer.compress(x, emb)
        end = time.perf_counter()
        time_list[1] = (end - start) * 1000
        time_list[2] = compression_time
        end_time = time.perf_counter()
        time_list[3] = (end_time - start_time) * 1000
        return compressed_obj, time_list

    @torch.no_grad()
    def receive(self, compressed_obj):
        emb = self.embedding_block(self.bpp_lmb_vec)
        feature = self.bottleneck_layer.decompress(compressed_obj, emb)
        # x2 = self.layer2[2:4](feature)
        x2 = self.layer2(feature)
        # x3 = self.layer3[3:6](feature)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feature = self.avgpool(x4)
        feature = torch.flatten(feature, 1)
        p_logits = self.fc(feature)
        return p_logits


@register_model
def matsubara2022wacv(num_classes=1000, bpp_lmb=1.28, teacher=True, mode='joint'):
    """ Supervised Compression for Resource-Constrained Edge Computing Systems

    - Paper: https://arxiv.org/abs/2108.11898
    - Github: https://github.com/yoshitomo-matsubara/supervised-compression

    Args:
        num_classes (int, optional): _description_. Defaults to 1000.
        bpp_lmb (float, optional): _description_. Defaults to 1.28.
        teacher (bool, optional): _description_. Defaults to True.
        mode (str, optional): _description_. Defaults to 'joint'.

    Returns:
        _type_: _description_
    """
    model = BottleneckResNet(24, num_classes=num_classes, bpp_lmb=bpp_lmb, teacher=teacher, mode=mode)
    return model
