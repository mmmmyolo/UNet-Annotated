import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

#原始光学图像
dir_img = Path('./data/imgs/')
#原始标签
dir_mask = Path('./data/masks/')
#保存权重路径
dir_checkpoint = Path('./checkpoints/')


def train_net(net,#网络定义
              device,#设置GPU还是CPU训练
              epochs: int = 5,#训练的轮次
              batch_size: int = 8,#一次训练所抓取的图片的数量
              learning_rate: float = 1e-5,#学习率  通过损失函数调整权重的快慢
              val_percent: float = 0.1,#验证集所占的比例
              save_checkpoint: bool = True,#是否保存权重文件，默认保存
              img_scale: float = 0.5,#图像的缩小因子
              amp: bool = False):#是否使用混合精度计算
    # 1. Create dataset   数据集的创建
    try:
        #跳转data_loading.py中的数据集构建类
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    #捕捉异常函数 当发生AssertionError, RuntimeError执行下列代码
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions  划分训练集与验证集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    #随机的划分dataset,长度为[n_train,n_val],torch.Generator().manual_seed(0)为随机数种子
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders  创建数据加载器
    #创建一个字典，类似于c++中的哈希表  num_workers表示是否多进程读取数据  pin_memory为True会将数据放置在GPU上
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    #https://blog.csdn.net/weixin_43981621/article/details/119685671
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging) 初始化日志
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #设置优化器，损失函数，学习率调度器
    #对反向传播的偏导数进行优化的优化器   通常偏导数不会直接作用于可学习参数
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #https://blog.csdn.net/u011622208/article/details/86574291
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    #设置混合精度训练
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    global_step = 0

    # 5. Begin training  训练部分
    for epoch in range(1, epochs+1):#epoch从1到{epoch}，训练了epoch次
        # 如果有Batch Normalization 和 Dropout  https://blog.csdn.net/weixin_44023658/article/details/105844861
        net.train()
        epoch_loss = 0
        #tqdm返回一个迭代器给变量pbar  https://blog.csdn.net/CSDN_OWL/article/details/114335467
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                #获取image和mask的张量
                images = batch['image']
                true_masks = batch['mask']

                #判断网络通道数与输入通道数是否相等
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                #将images和true_masks放在GPU上，返回值为正确设置了device的Tensor
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                #是否启用混合精度训练
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    #损失函数
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                #清空过往梯度值
                optimizer.zero_grad(set_to_none=True)
                #反向传播计算梯度值
                grad_scaler.scale(loss).backward()
                #通过梯度值更新权重
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                #全局训练步长
                global_step += 1
                #更新一个epoch中的累计损失
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round  验证环节
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    #每经过一个division_step进行一次for循环处理
                    if global_step % division_step == 0:
                        histograms = {}
                        #tag和value分别为网络每一层的名称和参数迭代器
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
        #每个epoch保存一次权重文件
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

#网络参数的设定
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    #初始化日志文件  https://blog.csdn.net/colinlee19860724/article/details/90965100
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #设置GPU还是cpu训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    #将网络放置在device上
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    #如果发生键盘中断
    except KeyboardInterrupt:
        #保存模型的权重参数
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
