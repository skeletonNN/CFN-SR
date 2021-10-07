
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from modules.transformer import TransformerEncoder

import sys
sys.path.append('..')
from MSAF import MSAF

class CrossModalAttentionLayer(nn.Module):
    # y attends x
    def __init__(self, k, x_channels, y_size, spatial=True):
        super(CrossModalAttentionLayer, self).__init__()
        self.k = k
        self.spatial = spatial

        if spatial:
            self.channel_affine = nn.Linear(x_channels, k)

        self.y_affine = nn.Linear(y_size, k, bias=False)

        self.audio_attention = TransformerEncoder(embed_dim=k,
                            num_heads=8,
                            layers=5,
                            attn_dropout=0.1,
                            relu_dropout=0.1,
                            res_dropout=0.1,
                            embed_dropout=0.3,
                            attn_mask=True)
        self.attn_weight_affine = nn.Linear(k, 1)

    def forward(self, x, y):
        # x -> [(S, C, H, W)], len(x) = bs
        # y -> (bs, D)

        # print(x.size(), y.size())

        original_y = y
        h_vs = self.audio_attention(y.permute(2, 0, 1))
        if type(h_vs) == tuple:
            h_vs = h_vs[0]

        last_vs = h_vs[-1]
        bs = y.size(0)
        y_k = self.y_affine(last_vs) # (bs, k)

        all_spatial_attn_weights_softmax = []

        for i in range(bs):
            if self.spatial:
                x_tensor = x[i].permute(1, 2, 3, 0) # (S_v, H_v, W_v, C_v)
                x_k = self.channel_affine(x_tensor) # (S_v, H_v, W_v, k)
                x_k += y_k[i]
                x_k = torch.tanh(x_k)
                x_attn_weights = self.attn_weight_affine(x_k).squeeze(-1) # (S_v, H_v, W_v)

                all_spatial_attn_weights_softmax.append(
                    F.softmax(x_attn_weights.reshape(x_tensor.size(0), -1),dim=-1).reshape(x_tensor.size(0), x_tensor.size(1), x_tensor.size(2)) # (S_v, H_v, W_v)
                )

        res = torch.zeros((x[0].permute(1, 2, 3, 0).size(0), x[0].permute(1, 2, 3, 0).size(1), x[0].permute(1, 2, 3, 0).size(2))).cuda()

        for i in all_spatial_attn_weights_softmax:
            res += i

        result = x*res
        # result += x
        return [result, original_y]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MSAFNet(nn.Module):
    def __init__(self, model_param):
        super(MSAFNet, self).__init__()
        # The inputs to these layers will be passed through msaf before being passed into the layer
        self.msaf_locations = {
            # "video": [6, 7],
            # "audio": [5, 11],
            "video": [7],
            "audio": [11],
        }
        # MSAF blocks
        self.msaf = nn.ModuleList([
            # MSAF(in_channels=[1024, 32], block_channel=16, block_dropout=0.2, reduction_factor=4),
            # CrossModalAttentionLayer(k=32, x_channels=1024, y_size=32, spatial=True),
            # MSAF(in_channels=[2048, 64], block_channel=32, block_dropout=0.2, reduction_factor=4)
            CrossModalAttentionLayer(k=64, x_channels=2048, y_size=64, spatial=True)
        ])
        self.num_msaf = len(self.msaf)

        self.fc = nn.Linear(3712, 8)

        if "video" in model_param:
            video_model = model_param["video"]["model"]
            # video model layers
            video_model = nn.Sequential(
                video_model.conv1,  # 0
                video_model.bn1,  # 1
                video_model.maxpool,  # 2
                video_model.layer1,  # 3
                video_model.layer2,  # 4
                video_model.layer3,  # 5
                video_model.layer4,  # 6
                video_model.avgpool,  # 7
                Flatten(),  # 8
                # video_model.fc  # 9
            )
            self.video_model_blocks = self.make_blocks(video_model, self.msaf_locations["video"])
            self.video_id = model_param["video"]["id"]

        if "audio" in model_param:
            audio_model = model_param["audio"]["model"]
            # audio model layers
            audio_model = nn.Sequential(
                audio_model.conv1,  # 0
                nn.ReLU(inplace=True),  # 1
                audio_model.bn1,  # 2
                audio_model.conv2,  # 3
                nn.ReLU(inplace=True),  # 4
                audio_model.maxpool,  # 5
                audio_model.bn2,  # 6
                audio_model.dropout1,  # 7
                audio_model.conv3,  # 8
                nn.ReLU(inplace=True),  # 9
                audio_model.bn3,  # 10
                audio_model.flatten,  # 11
                audio_model.dropout2,  # 12
                # audio_model.fc1  # 13
            )
            self.audio_model_blocks = self.make_blocks(audio_model, self.msaf_locations["audio"])
            self.audio_id = model_param["audio"]["id"]

    def forward(self, x):
        for i in range(self.num_msaf + 1):
            if hasattr(self, "video_id"):
                x[self.video_id] = self.video_model_blocks[i](x[self.video_id])
            if hasattr(self, "audio_id"):
                x[self.audio_id] = self.audio_model_blocks[i](x[self.audio_id])
            if i < self.num_msaf:
                x = self.msaf[i](x[0], x[1])
                # if i == 0:
                #     x = self.msaf[i](x)
                # else:
                #     x = self.msaf[i](x[0], x[1])
                # x = self.msaf[i](x)

        res = torch.cat(x, dim=1)
        res = self.fc(res)
                
        return res

    # split model into blocks for msafs. Model in Sequential. recipe in list
    def make_blocks(self, model, recipe):
        blocks = [nn.Sequential(*(list(model.children())[i:j])) for i, j in zip([0] + recipe, recipe + [None])]
        return nn.ModuleList(blocks)
