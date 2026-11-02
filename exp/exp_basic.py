import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # 只有在没有外部设置CUDA_VISIBLE_DEVICES时才设置
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            
            # 使用第一个可见的GPU
            device = torch.device('cuda:0')
            print(f'Use GPU: cuda:0 (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")})')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
