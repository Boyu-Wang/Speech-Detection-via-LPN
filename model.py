# model for sequence classify

import torch
import torch.nn as nn

class AppearanceModel(nn.Module):
    def __init__(self, num_class, dropout):
        super(AppearanceModel, self).__init__()
        self.dropout = dropout
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(20*20*64, 64),
        )
        self.recurrent_layer = nn.GRU(64, 64, 2, batch_first=False, dropout=dropout)
        self.project_layer = nn.Linear(64, num_class)

    # the size of input is [batch_size, seq_len, 3,  H, W]
    # the size of input_landmark is [batch_size, seq_len, num_landmark, 2]
    def forward(self, input, input_landmark):
        batch_size, seq_len, _, H, W = input.size()
        input = input.view([batch_size*seq_len, 3, H, W])
        input_fea = self.cnn_layer(input)
        input_fea = input_fea.view([batch_size*seq_len, -1])
        input_fea = self.fc_layer(input_fea)
        input_fea = input_fea.view([batch_size, seq_len, -1])
        rnn_outputs, _ = self.recurrent_layer(input_fea) # [B, L, D]

        logits = self.project_layer(rnn_outputs) # [B, L, C]

        avg_logits = logits.mean(1)
        last_logits = logits[:, -1, :]
        return avg_logits, last_logits



class LPN(nn.Module):
    def __init__(self, num_class, dropout):
        super(LPN, self).__init__()
        self.dropout = dropout
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(20*64, 64),
            )
        self.recurrent_layer = nn.GRU(64, 64, 2, batch_first=False, dropout=dropout)
        self.project_layer = nn.Linear(64, num_class)

    # the size of input_img is [batch_size, seq_len, 3,  H, W]
    # the size of input_landmark is [batch_size, seq_len, num_landmark, 2]
    def forward(self, input_img, input_landmark):
        batch_size, seq_len, _, H, W = input_img.size()
        num_landmark = input_landmark.size(2)
        input_img = input_img.view([batch_size*seq_len, 3, H, W])
        input_img_fea = self.cnn_layer(input_img) # [batch_size, seq_len, C, 20, 20]
        input_img_fea = input_img_fea.view([batch_size * seq_len, -1, H//2, W//2])
        input_img_fea = input_img_fea.permute([0, 2, 3, 1]) # [B*L, 20, 20, C]
        # landmark pooling
        input_landmark = input_landmark.view([batch_size*seq_len, num_landmark, 2]) # [B*L, num_landmark, 2]

        pooled_img_fea = input_img_fea[torch.arange(batch_size*seq_len).unsqueeze(1), input_landmark[:, :, 0], input_landmark[:,:,1]] #[B*L, num_landmark, C]
        pooled_img_fea = pooled_img_fea.view([batch_size * seq_len, -1])

        input_fea = self.fc_layer(pooled_img_fea)
        input_fea = input_fea.view([batch_size, seq_len, -1])
        rnn_outputs, _ = self.recurrent_layer(input_fea) # [B, L, D]

        logits = self.project_layer(rnn_outputs) # [B, L, 2]

        avg_logits = logits.mean(1)
        last_logits = logits[:, -1, :]
        return avg_logits, last_logits


class AppearanceLPNModel(nn.Module):
    def __init__(self, num_class, dropout):
        super(AppearanceLPNModel, self).__init__()
        self.dropout = dropout
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
        )
        self.fc_layer_1 = nn.Sequential(
            nn.Linear(20*20*64, 64),
            )
        self.fc_layer_2 = nn.Sequential(
            nn.Linear(20 * 64, 64),
        )
        self.recurrent_layer = nn.GRU(64*2, 64, 2, batch_first=False, dropout=dropout)
        self.project_layer = nn.Linear(64, num_class)

    # the size of input_img is [batch_size, seq_len, 3,  H, W]
    # the size of input_landmark is [batch_size, seq_len, num_landmark, 2]
    def forward(self, input_img, input_landmark):
        batch_size, seq_len, _, H, W = input_img.size()
        num_landmark = input_landmark.size(2)
        input_img = input_img.view([batch_size*seq_len, 3, H, W])
        input_img_fea = self.cnn_layer(input_img) # [batch_size, seq_len, C, 20, 20]
        input_img_fea = input_img_fea.view([batch_size * seq_len, -1, H//2, W//2])
        input_img_fea = input_img_fea.permute([0, 2, 3, 1]).contiguous() # [B*L, 20, 20, C]

        ori_input_fea = input_img_fea.view([batch_size*seq_len, -1])
        ori_input_fea = self.fc_layer_1(ori_input_fea)
        ori_input_fea = ori_input_fea.view([batch_size, seq_len, -1])

        # landmark pooling
        input_landmark = input_landmark.view([batch_size*seq_len, num_landmark, 2]) # [B*L, num_landmark, 2]
        pooled_img_fea = input_img_fea[torch.arange(batch_size*seq_len).unsqueeze(1), input_landmark[:, :, 0], input_landmark[:,:,1]] #[B*L, num_landmark, C]
        pooled_img_fea = pooled_img_fea.view([batch_size * seq_len, -1])

        pooled_input_fea = self.fc_layer_2(pooled_img_fea)
        pooled_input_fea = pooled_input_fea.view([batch_size, seq_len, -1])

        input_fea = torch.cat([ori_input_fea, pooled_input_fea], 2)
        rnn_outputs, _ = self.recurrent_layer(input_fea) # [B, L, D]

        logits = self.project_layer(rnn_outputs) # [B, L, 2]

        avg_logits = logits.mean(1)
        last_logits = logits[:, -1, :]
        return avg_logits, last_logits
