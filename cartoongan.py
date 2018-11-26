from torch import nn
import visdom

from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU, ReLU, BatchNorm2d


class Generator(nn.Module):

    def __init__(self, in_dim=3):

        super(Generator, self).__init__()

        self.down = nn.Sequential(
        #k7n64s1  out H x W
        Conv2d(kernel_size=7, in_channels=in_dim, out_channels=64, stride=1, padding=3, bias=False),
        BatchNorm2d(64),
        ReLU(),

        #Down-convolution
        #k3n128s2, k3n128s1   out H/2 x W/2
        Conv2d(kernel_size=3, in_channels=64, out_channels=128, stride=2, padding=1, bias=False),
        Conv2d(kernel_size=3, in_channels=128, out_channels=128, stride=1, padding=1, bias=False),
        BatchNorm2d(128),
        ReLU(),

        # k3n256s2, k3n256s1 out H/4 x W/4
        Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=2, padding=1, bias=False),
        Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
        BatchNorm2d(256),
        ReLU() )

        self.res1 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256) )

        self.res2 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res3 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res4 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res5 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res6 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res7 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))

        self.res8 = nn.Sequential(
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256))


        self.up = nn.Sequential(
            #up-convolution  out H/2 x W/2
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            Conv2d(kernel_size=3, in_channels=128, out_channels=128, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(),

            # out H x W
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1,
                            bias=False),
            Conv2d(kernel_size=3, in_channels=64, out_channels=64, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(),

            # Final conv
            Conv2d(kernel_size=7, in_channels=64, out_channels=3, stride=1, padding=3, bias=False),
            nn.Tanh())


    def forward(self, x):

        d = self.down(x)

        res1 = self.res1(d)

        res1 = res1 + d

        res2 = self.res2(res1)

        res2 = res1 + res2

        res3 = self.res3(res2)

        res3 = res3 + res2

        res4 = self.res4(res3)

        res4 = res3 + res4

        res5 = self.res5(res4)

        res5 = res4 + res5

        res6 = self.res6(res5)

        res6 = res5 + res6

        res7 = self.res7(res6)

        res7 = res6 + res7

        res8 = self.res8(res7)

        res8 = res7 + res8

        out = self.up(res8)

        return out

class discriminator(nn.Module):

    def __init__(self, in_dim):

        super(discriminator, self).__init__()

        self.dis = nn.Sequential(

            Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            LeakyReLU(0.2,True),

            Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            LeakyReLU(0.2,True),
            Conv2d(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2,True),

            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, True),
            Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, True),

            Conv2d(kernel_size=3, in_channels=256, out_channels=256, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, True),

            Conv2d(kernel_size=3, in_channels=256, out_channels=1, stride=1, padding=1, bias=False) )


    def forward(self, x):

        out = self.dis(x)

        return out





