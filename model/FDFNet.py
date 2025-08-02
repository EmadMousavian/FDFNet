import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):

        super(BasicBlock, self).__init__()
        self.residual_only = residual_only
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and '
                             'base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.act = activation
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual_only:
            return out
        out = out + identity
        out = self.act(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


class DCMAF(nn.Module):
    def __init__(self, num_channel, out_channel):
        super().__init__()

        # todo add convolution here
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]

        self.conv1 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, image, filter_img):
        image = self.pool(image)
        filter_img = self.pool(filter_img)

        image = F.relu(self.conv1(image))
        filter_img = F.relu(self.conv2(filter_img))

        image = self.conv3(image)
        filter_img = self.conv4(filter_img)

        diff = image - filter_img
        weight = self.activation(diff)
        w1 = weight
        w2 = 1 - weight
        return w1, w2


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, pretrained_url=None, mode='train'):

        super(ResNet, self).__init__()
        # orginal image
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # filtered image
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        #         self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        ##### dcmaf fusion
        self.fusion_2 = DCMAF(64, 32)
        self.fusion_3 = DCMAF(128, 64)
        self.fusion_4 = DCMAF(256, 128)
        self.fusion_5 = DCMAF(512, 256)
        ######

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes)
        )

        if isinstance(pretrained_url, str) and mode == 'train':
            self.pretrained_url = pretrained_url
            self._load_resnet_pretrained()
            print("********************** Pretrained model loaded **********************")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _load_resnet_pretrained(self):
        # pretrain_dict = torch.load(self.pretrained_path)
        pretrain_dict = model_zoo.load_url(self.pretrained_url)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = torch.mean(v, 1).data. \
                        view_as(state_dict[k])
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, x, f):
        out = F.relu(self.bn1(self.conv1(x)))
        out_f = F.relu(self.bn1_d(self.conv1_d(f)))

        out = self.layer1(out)
        out_f = self.layer1_d(out_f)

        w_i, w_f = self.fusion_2(out, out_f)
        img_w = out.mul(w_i)
        fil_w = out_f.mul(w_f)
        out = out + fil_w
        out_f = out_f + img_w
        ###########

        out = self.layer2(out)
        out_f = self.layer2_d(out_f)

        w_i, w_f = self.fusion_3(out, out_f)
        img_w = out.mul(w_i)
        fil_w = out_f.mul(w_f)
        out = out + fil_w
        out_f = out_f + img_w
        ###########

        out = self.layer3(out)
        out_f = self.layer3_d(out_f)

        w_i, w_f = self.fusion_4(out, out_f)
        img_w = out.mul(w_i)
        fil_w = out_f.mul(w_f)
        out = out + fil_w
        out_f = out_f + img_w
        ###########

        out = self.layer4(out)
        out_f = self.layer4_d(out_f)

        w_i, w_f = self.fusion_5(out, out_f)
        img_w = out.mul(w_i)
        fil_w = out_f.mul(w_f)
        out = img_w + fil_w
        ##########

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def build_network(model_cfg, mode='train'):
    model = ResNet(
        block=BasicBlock, layers=model_cfg.LAYERS , num_classes=model_cfg.NUM_CLASSES,
        pretrained_url=model_urls[model_cfg.NAME], mode=mode
    )
    return model
