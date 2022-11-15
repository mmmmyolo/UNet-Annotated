import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    #不启用 BatchNormalization 和 Dropout
    net.eval()
    #将数组转化成张量
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    #在0的位置增加一个维度
    img = img.unsqueeze(0)
    #将img放在device上
    img = img.to(device=device, dtype=torch.float32)

    #torch.no_grad()不进行反向传播，不对网络进行更新，单纯的预测
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            #将值映射到0-1之间
            probs = torch.sigmoid(output)[0]

        #整合了先转化成图片，然后重新设定尺寸，再转化成张量的步骤
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        #对图片进行预测
        full_mask = tf(probs.cpu()).squeeze()

    #如果只有背景和一类目标
    if net.n_classes == 1:
        #返回值大于out_threshold的部分
        return (full_mask > out_threshold).numpy()
    else:
        #返回one_hot格式
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

#测试所需参数列表
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='F:\Pproject\Pytorch-UNet-master\checkpoints\checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    #构建参数列表
    args = get_args()
    #构建输入图片路径
    in_files = args.input
    #输出图片路径
    out_files = get_output_filenames(args)

    #实例化网络结构
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    #设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #info级别更新日志
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    #将网络放在device上
    net.to(device=device)
    #加载模型权重
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        #原始图片
        img = Image.open(filename)
        #预测图片
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        #保存预测图片
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
        #处理图片的时候可视化
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
