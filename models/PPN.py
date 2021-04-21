from .resnet.resnet import resnet50, resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch
import time

class local_rein_classifier(nn.Module):
    def __init__(self, numClass):
        super(local_rein_classifier, self).__init__()
        self.m = 1
        # resnet18
        self.backbone = resnet18(True)
        self.linear = nn.Sequential(
            nn.Linear(256, 1),
        )

        for n, m in self.named_children():
            if n == 'backbone':
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, patch_n):
        # resnet18
        _, _, x, _ = self.backbone(x)
        b, c, h, w = x.size()
        x1 = F.avg_pool2d(x, h).view(b, -1)
        x1 = self.linear(x1)
        #x1 = torch.sigmoid(x1)

        x2 = F.avg_pool2d(x, (h // patch_n[0], w // patch_n[1]))

        pes = []
        for i in range(patch_n[0]):
            for j in range(patch_n[1]):
                p = x2[:, :, i, j]
                p = self.linear(p)
                #p = torch.sigmoid(p)
                p = (x1 - p) * self.m
                p = torch.sigmoid(p)
                pes.append(p)

        x = torch.cat(pes, dim=-1)
        return x


class se_layer(nn.Module):
    def __init__(self,  numClass, reduction=1):
        super(se_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(numClass, numClass // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(numClass // reduction, numClass, bias=False),
            nn.Sigmoid()
        )
        for n, m in self.named_children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class fpn_module_global(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_global, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # global branch
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)

        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        out = torch.cat([p5, p4, p3, p2], dim=1)
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, global_=False, patch=False):
        # global
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)
        return output

class fpn_module_local(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_local, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # global branch
        # Top layer
        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(32*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)

        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        out = torch.cat([p5, p4, p3, p2], dim=1)
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, global_=False, patch=False):
        # global
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)
        return output

class rein_output_layer(nn.Module):
    def __init__(self, numClass):
        super(rein_output_layer, self).__init__()
        self.conv_layers = []
        self.n_class = numClass
        for i in range(numClass):
            l = nn.Sequential(
                nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, stride=1)
            )

            self.add_module('layer_class_%d' % i, l)
            self.conv_layers.append(l)
        # self.agg_conv = nn.Sequential(
        #         nn.Conv2d(numClass, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(64, numClass, kernel_size=1, stride=1)
        # )

    def forward(self, x1, x2):
        temp = []
        for i in range(self.n_class):
            temp.append(self.conv_layers[i](torch.cat([x1[:, i:i+1, :, :], x2[:, i:i+1, :, :]], dim=1)))
        x = torch.cat(temp, dim=1)
        # x = F.softmax(x, dim=1)
        # x = self.agg_conv(x)
        return x

class adaLG(nn.Module):
    def __init__(self, numClass):
        super(adaLG, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        self.max_patch_side = 960
        # Res net
        self.resnet_g = resnet50(True)
        self.resnet_l = resnet18(True)
        # fpn module
        self.fpn_g = fpn_module_global(numClass)
        self.fpn_l = fpn_module_local(numClass)
        # classifier
        self.rein_classifier = local_rein_classifier(numClass)
        self.inference_time = 0
        self.saliency_classifier = nn.Sequential(nn.Conv2d(numClass, numClass, kernel_size=3, stride=1, padding=1),
                                      # nn.BatchNorm2d(numClass),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(numClass, 3, kernel_size=5, stride=1, padding=2),
                                      )

        self.saliency = nn.Sequential(nn.Conv2d(numClass, numClass, kernel_size=3, stride=1, padding=1),
                                      # nn.BatchNorm2d(numClass),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(numClass, 3, kernel_size=5, stride=1, padding=2),
                                      # nn.Conv2d(numClass, 3, kernel_size=1, stride=1),
                                      )

        #self.rein_output_layer = rein_output_layer(numClass)
        self.rein_output_layer = nn.Sequential(
            # nn.BatchNorm2d(numClass * 2),
            nn.Conv2d(numClass * 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.3),
            # nn.ReLU(inplace=True),
            # se_layer(64, reduction=16),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.Dropout2d(0.5),
            # nn.ReLU(inplace=True),
            # se_layer(64, reduction=8),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # # # nn.Dropout2d(0.1),
            # # nn.ReLU(inplace=True),
            # # # se_layer(32, reduction=4),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            # # # nn.Dropout2d(0.1),
            # nn.ReLU(inplace=True),
            # se_layer(32, reduction=2),
            nn.Dropout(0.1),
            nn.Conv2d(256, numClass, kernel_size=1, stride=1)
        )

        # init rein_output_layer
        for n, m in self.rein_output_layer.named_modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.kaiming_normal_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
                # nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1., 0.02)
                nn.init.constant_(m.bias.data, 0.)

        # init fpn
        for m in self.fpn_g.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

        # init fpn
        for m in self.fpn_l.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)

        # init saliency
        for m in self.saliency_classifier.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)

        for m in self.saliency.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)


    def get_inference_time(self):
        return self.inference_time

    def forward(self, inputs, patch_inp=None, template=None, patch_n=None, mode=1, glb_res=None):
        torch.cuda.synchronize()
        start = time.time()
        if mode == 0:
            # train or predict from local model
            c2_g, c3_g, c4_g, c5_g = self.resnet_l.forward(inputs)
            output_l = self.fpn_l.forward(c2_g, c3_g, c4_g, c5_g)
            torch.cuda.synchronize()
            end = time.time()
            self.inference_time += (end - start)
            return output_l

        elif mode == 1:
            # train or predict from global
            c2_g, c3_g, c4_g, c5_g = self.resnet_g.forward(inputs)
            output_g = self.fpn_g.forward(c2_g, c3_g, c4_g, c5_g)
            # imsize = image_global.size()[2:]
            # output_g = F.interpolate(output_g, imsize, mode='nearest')
            torch.cuda.synchronize()
            end = time.time()

            self.inference_time += (end - start)
            return output_g

        elif mode == 2:
            if glb_res is not None:
                x = F.interpolate(glb_res, size=inputs.size()[2:], mode='bilinear', align_corners=True)
                x = self.saliency_classifier(F.softmax(x, dim=1))
                inputs = x * inputs
            # train or predict from local reinforce classifier
            torch.cuda.synchronize()
            end = time.time()
            self.inference_time += (end - start)
            return self.rein_classifier.forward(inputs, patch_n)

        else:
            if template is None:
                # patch reinforcement
                x = self.saliency(inputs)
                x = patch_inp * x
                # x = patch_inp

                c2_g, c3_g, c4_g, c5_g = self.resnet_l.forward(x)
                x = self.fpn_l.forward(c2_g, c3_g, c4_g, c5_g)

            else:
                # global reinforcement
                # x = inputs + template
                # x = torch.sum([inputs, template], dim=1)
                # x = torch.cat([inputs, template], dim=1)
                x = torch.cat([F.softmax(inputs, dim=1), F.softmax(template, dim=1)], dim=1)
                x = self.rein_output_layer(x)
                # x = torch.cat([F.relu(inputs), F.relu(template)], dim=1)
                # x = self.rein_output_layer(F.softmax(inputs, dim=1), F.softmax(template, dim=1))
                # x = self.rein_output_layer(F.softmax(inputs, dim=1), F.softmax(template, dim=1))
            torch.cuda.synchronize()
            end = time.time()
            # t = (end - start)
            self.inference_time += (end - start)
            return x


