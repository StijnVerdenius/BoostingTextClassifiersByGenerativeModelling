import torch.nn as nn
import torch
from models.GeneralModel import GeneralModel


class CrossEntropyLoss(GeneralModel):

    def __init__(self, n_channels_in=(0), device="cpu", **kwargs):
        super(CrossEntropyLoss, self).__init__(n_channels_in, device, **kwargs)
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, target, out_data):
        # out_data = out_data.permute([1, 0, -1])

        # # todo delete after checking w dummy spam data
        # target_onehot = torch.zeros((target.shape[0], 2))
        # for i, val in enumerate(target):
        #     target_onehot[i][val] = 1
        # # todo till here
        # print(out_data.shape)
        # print(out_data)
        # print(target_onehot.shape)
        # print(target_onehot)
        return self.loss(out_data, target.cpu())
