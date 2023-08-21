import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config

        if self.config['pretrained']:
            net = getattr(models, self.config['network'])(weights='IMAGENET1K_V1')
        else:
            net = getattr(models, self.config['network'])()

        if 'resnet' in self.config['network']:
            in_features = list(net.fc.modules())[-1].in_features
        else:
            raise ValueError('Unknown backbone')

        modules = list(net.children())
        if 'resnet' in self.config['network']:
            modules = modules[:-2]
        else:
            raise ValueError('Unknown backbone')

        self.net = nn.Sequential(*modules)

        # classification module
        self.classifiers = nn.ModuleDict()
        for t_name in self.config['cls_tasks']:
            modules = list()
            modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            modules.append(nn.Flatten())
            modules.append(nn.Linear(in_features, self.config['cls_tasks'][t_name]))
            classifier = nn.Sequential(*modules)
            self.classifiers[t_name] = classifier

        self.gen_cam_map = self.config['gen_cam_map']

    def forward(self, input):
        features = self.net(input)
        cls_logits = list()
        cls_maps = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]

            # get CAM maps
            c = self.get_cam_faster(features, classifier)
            cls_maps.append(c)

            if len(self.config['attn_tasks']) != 0:
                # attention guidance computes soft weights rather than AVG
                a = torch.softmax(c.reshape(c.shape[0], c.shape[1], -1), dim=2).reshape(c.shape)
                cls_logits.append((c.contiguous() * a).sum(dim=(2, 3)))
            else:
                # AVG
                cls_logits.append(c.mean(dim=(2, 3)))

        attn_logits = list()  # []
        if len(self.config['attn_tasks']) != 0:
            for a_name in self.config['attn_tasks']:
                c_name = self.config['attn_tasks'][a_name]['attn_of']
                cls_maps_idx = list(self.config['cls_tasks'].keys()).index(c_name)
                attn_logits.append(cls_maps[cls_maps_idx])

        return cls_logits, attn_logits, cls_maps

    def get_cam_faster(self, features, classifier):

        if not self.gen_cam_map:
            return None

        cls_weights = classifier[-1].weight
        cls_bias = classifier[-1].bias

        act_maps = F.conv2d(features, cls_weights.view(cls_weights.shape[0], cls_weights.shape[1], 1, 1),
                            cls_bias, stride=1, padding=0, dilation=1, groups=1)

        return act_maps


def get_network(config):
    return Net(config)
