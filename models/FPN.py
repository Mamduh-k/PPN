import torch.nn as nn
import torch.nn.functional as F
import torch

class fpn_module_global(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_global, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # global branch
        # Top layer
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # out_channels=256)


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
        self.numclass = numClass

        self.aux_head = nn.Sequential(
            nn.Conv2d(512, 512,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=512),
            nn.Conv2d(512, self.numclass,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

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

    def forward(self, c2, c3, c4, c5, c5_l2g=None):


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
        return output,ps3

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

        self.g2l_c5 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        # self.ps3_up = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.up_regular = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
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

    def forward(self, c2, c3, c4, c5, c5_g2l=None):
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
        return output, ps3

