import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

#Dataset为pytorch官方给出的数据构建方法，继承时需要重写__len__()方法：获取数据集的大小,__getitem__()方法：获取数据集中任意一个元素
class BasicDataset(Dataset):
    #:为建议传入参数的类型   函数后的箭头->是建议的返回值类型
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        #Path方法 获取数据集路径
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        #判断图片的缩放因子必须在o-1之间
        assert 0 < scale <= 1,'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        #listdir获取数据集目录中所有文件成一个列表
        #file.startswith('.')函数表示以.开头的文件
        #splitext(file)[0]分离文件名与后缀名得到除去后缀的名称
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        #记录日志信息
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    #获取数据集长度方法
    def __len__(self):
        return len(self.ids)

    #@staticmethod类似于c++中的static关键字，表示该类中的方法为静态成员方法，该方法的调用可不依赖对象，所以不含self指针
    #相当于类内封装类外方法
    @staticmethod
    #将数据集通过scale进行缩放
    def preprocess(pil_img, scale, is_mask):
        #获取图片的宽和高
        w, h = pil_img.size
        #缩放scale倍
        newW, newH = int(scale * w), int(scale * h)
        #判断新的图片像素必须大于0
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #对图片进行缩放  如果是标签文件:Image.NEAREST低质量  不是标签文件Image.BICUBIC三次样条插值
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        #将pil_img转化成数组  520*720的图片读出来为(720,520,3)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            #如果是二维的数组数据
            if img_ndarray.ndim == 2:
                #np.newaxis增加一个维度
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                #transpose(2, 0, 1)，把数组变为第二个参数变为第0个参数，第0个参数变为第一个，第一个参数变为第二个
                #按照上面注释的图片尺寸，transpose后为(3,720,520),可以成功将rgb图片三个通道分离出来
                #相当于3个720*520的数组
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            #https://blog.csdn.net/sdlyjzh/article/details/8245145
            img_ndarray = img_ndarray / 255

        return img_ndarray

    #文件加载函数
    @staticmethod
    def load(filename):
        #获取分离后图片格式的后缀名
        ext = splitext(filename)[1]
        if ext == '.npy':
            #np.load读磁盘数组数据的函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中
            #Image.fromarray将数组转化成图片
            return Image.fromarray(np.load(filename))
        #如果是权重文件
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            #如果是图片直接打开
            return Image.open(filename)
    #获取数据集中任意元素
    def __getitem__(self, idx):
        #获取不带后缀的图片名称
        name = self.ids[idx]
        #分别在原始图片和标签中寻找同名文件
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        #判断是否找到文件,如果生成列表长度不等于1则报错
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        #调用load方法加载图片
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        #判断训练图片与标签图片的尺寸是否一致
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #将训练图片与标签图片通过preprocess方法进行缩放
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        #将image与mask转化成张量返回，数据类型为float,contiguous()表示拷贝一份新的数据，类似于深拷贝
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    #CarvanaDataset的__init__函数，实例化时会调用该函数
    #self相当于c++中的this指针
    def __init__(self, images_dir, masks_dir, scale=1):
        #BasicDataset类的构造函数，使用super()方法进行调用
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')

