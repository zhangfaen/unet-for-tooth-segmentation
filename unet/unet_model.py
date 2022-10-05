""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import logging

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # logging.info("input shape:" + str(x.shape))
        # logging.info("# of values >= 1 in x:" + str(torch.sum(x >= 1.0)))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # logging.info("# of values >= 1 in x5:" + str(torch.sum(x5 >= 1.0)))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logging.info("# of values >= 1 after up4:" + str(torch.sum(x >= 1.0)))
        logits = self.outc(x)
        # logging.info("output shape:" + str(logits.shape))
        # logging.info("# of values >= 1 in logit:" + str(torch.sum(logits >= 1.0)))

        # (N1, C1, H1, W1) of input x,  (N2, C2, H2, W2) of output(aka loggits),
        # assert H1 == H2, W1 == W2
        return logits
