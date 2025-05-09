import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import cleanlab
from tqdm import tqdm
from dataloaders.MSD_lung import (MSD_lung, CenterCrop, RandomCrop,
                             RandomRotFlip, ToTensor,
                             TwoStreamBatchSampler)
from dataloaders.la_heart import (LA_heart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                             RandomRotFlip, ToTensor,
                             TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/MSD_data_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MCL3S_MSD', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[160, 160, 120],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2025, help='random seed')

parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=51,
                    help='total samples')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')

parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = MSD_lung(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noisy_ema_inputs = unlabeled_volume_batch + noise
            ema_inputs = unlabeled_volume_batch

            outputs, pred_p, _ = model(volume_batch, turnoff_drop=False, feature_pert=True)
            pred_x, pred_u_p = outputs.split([args.labeled_bs, args.batch_size - args.labeled_bs])

            pred_u_p = pred_u_p.detach()
            conf_u_p = pred_u_p.softmax(dim=1).max(dim=1)[0]
            mask_u_p = pred_u_p.argmax(dim=1)

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_onehot = torch.argmax(outputs_soft, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                noisy_ema_output = ema_model(noisy_ema_inputs)
                noisy_ema_output_soft = torch.softmax(noisy_ema_output, dim=1)
                unlabeled_pred = outputs_onehot[args.labeled_bs:].cpu().detach().numpy()
                ema_soft = ema_output_soft.cpu().detach().numpy()
                ema_soft_transposed = np.transpose(ema_soft, (0, 2, 3, 4, 1))
                ema_soft_flat = ema_soft_transposed.reshape(-1, num_classes)
                ema_soft_flat = np.ascontiguousarray(ema_soft_flat)
                unlabeled_pred_flat = unlabeled_pred.reshape(-1).astype(np.uint8)
                assert unlabeled_pred_flat.shape[0] == ema_soft_flat.shape[0]
                current_consistency_weight = get_current_consistency_weight(iter_num // 150)
                mse_consistency = losses.softmax_mse_loss(outputs[args.labeled_bs:], noisy_ema_output)

                try:
                    if args.CL_type == 'both':
                        detected_errors = cleanlab.filter.find_label_issues(
                            unlabeled_pred_flat, ema_soft_flat, filter_by='both', n_jobs=1)
                    elif args.CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                        detected_errors = cleanlab.filter.find_label_issues(
                            unlabeled_pred_flat, ema_soft_flat, filter_by=args.CL_type, n_jobs=1)
                    error_map_np = detected_errors.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(
                        np.uint8)
                    error_map_tensor = torch.from_numpy(error_map_np).cuda(outputs_soft.device.index)
                    binary_mask = (error_map_tensor == 1).float().unsqueeze(1)
                    binary_mask = torch.cat([binary_mask, binary_mask], dim=1)
                    consistency_loss = torch.sum(binary_mask * mse_consistency) / (torch.sum(binary_mask) + 1e-16)
                except Exception as error:
                    consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - noisy_ema_output_soft) ** 2)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_p = dice_loss(pred_p.softmax(dim=1), mask_u_p.unsqueeze(1).float(),
                                         ignore=(conf_u_p < 0.95).float())

            supervised_loss = 0.5 * (loss_dice + loss_ce)
            loss = supervised_loss + current_consistency_weight * consistency_loss + 0.2 * loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_p', loss_p, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              current_consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_p: %f ' %  #
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_p.item()))  #
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 500 and iter_num % 50 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
