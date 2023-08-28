import torch.nn.functional as F
import torch
import torch.nn as nn

from .common import default_conv

class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride = 1):

        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample=None
        if stride!=1:
            self.downsample = nn.Conv2d(inplanes, planes, 1, stride)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ZeroPad2d(1), 
                    #   nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ZeroPad2d(1),
                    #   nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# small model version
class FeatureExtractorWithInfo(nn.Module):
    def __init__(self, n_residual_blocks, in_features):
        super(FeatureExtractorWithInfo, self).__init__()
        # Residual blocks
        rcab = True

        model_body_block1 = []
        model_body_block2 = []
        model_body_block3 = []
        model_body_block4 = []
        for _ in range(n_residual_blocks):
            
            model_body_block1 += [ResidualBlock(in_features)]
            model_body_block2 += [ResidualBlock(in_features)]
            model_body_block3 += [ResidualBlock(in_features)]
            model_body_block4 += [ResidualBlock(in_features)]

        self.model_body_block1 = nn.Sequential(*model_body_block1)
        self.model_body_block2 = nn.Sequential(*model_body_block2)
        self.model_body_block3 = nn.Sequential(*model_body_block3)
        self.model_body_block4 = nn.Sequential(*model_body_block4)

    def forward(self, x, info):
        x = self.model_body_block1(x) + info
        x = self.model_body_block2(x) + info
        x = self.model_body_block3(x) + info
        x = self.model_body_block4(x) + info
        return x

# small model version
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=8):

        super(Generator, self).__init__()

        # Initial convolution block
        model_head1 = [nn.ZeroPad2d(3),
                    #    nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]
        
        in_features = 64
        out_features = in_features * 2

        model_head2 = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features * 2

        model_head3 = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features * 2

        # Main blocks
        self.feature_extractor = FeatureExtractorWithInfo(n_residual_blocks//4, in_features)
        
        # auxiliary block
        num_classes=50
        model_body2 = []
        for _ in range(3):
            model_body2 += [BasicBlock(in_features, in_features)]
            model_body2 += [BasicBlock(in_features, in_features, stride=2)]

        self.model_body2 = nn.Sequential(*model_body2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Upsampling
        out_features = in_features // 2
        model_tail1 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features // 2
        model_tail2 = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features // 2
        model_tail3 = [nn.ZeroPad2d(3),
                    #    nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]
        

        self.model_head1 = nn.Sequential(*model_head1)
        self.model_head2 = nn.Sequential(*model_head2)
        self.model_head3 = nn.Sequential(*model_head3)

        self.model_tail1 = nn.Sequential(*model_tail1)
        self.model_tail2 = nn.Sequential(*model_tail2)
        self.model_tail3 = nn.Sequential(*model_tail3)

    def forward(self, x):
        x1 = self.model_head1(x)
        x2 = self.model_head2(x1)
        sfeat = self.model_head3(x2)

        info = self.model_body2(sfeat)
        info = self.avgpool(info)
        aux = torch.flatten(info, 1)
        aux = self.fc(aux)

        b, c, h, w = sfeat.shape
        info = info.expand([b,c,h,w])

        feat = self.feature_extractor(sfeat, info)
        
        out1 = self.model_tail1(feat) + x2
        out2 = self.model_tail2(out1) + x1
        out = self.model_tail3(out2)
        return out, aux
    
    def get_info_feature(self, x):
        x1 = self.model_head1(x)
        x2 = self.model_head2(x1)
        sfeat = self.model_head3(x2)

        info = self.model_body2(sfeat)
        info = self.avgpool(info)
        np_arr_info = torch.squeeze(info).data.cpu().numpy()
        return np_arr_info
    

# discriminator block
class Block(nn.Module): # conv > instancenorm > lrelu
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,# kernel
                stride,
                1,# padding
                bias=True,
                # padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                input_nc,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                # padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        input_nc = features[0]
        for feature in features[1:]:
            layers.append(
                Block(input_nc, feature, stride=1 if feature == features[-1] else 2)
            )
            input_nc = feature
        layers.append(
            nn.Conv2d(
                input_nc,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                # padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)
