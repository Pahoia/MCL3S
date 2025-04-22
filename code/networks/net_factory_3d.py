from networks.unet_3D import unet_3D
from networks.vnet import VNet, VNet_MTPD
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.pnet import PNet
# from networks.efficientunet import Effi_UNet

def net_factory_3d(net_type="vnet", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_MTPD":
        net = VNet_MTPD(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "pnet":
        net = PNet(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    else:
        net = None
    return net
