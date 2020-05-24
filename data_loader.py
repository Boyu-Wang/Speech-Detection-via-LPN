from PIL import Image
import torch.utils.data as data
import os
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


class LSW(data.Dataset):
    def __init__(self, root_path, subset, seq_len=20, load_landmark=False):

        self.img_root_path = os.path.join(root_path, 'aligned_mouth_img', subset)
        self.landmark_root_path = os.path.join(root_path, 'aligned_mouth_landmark', subset)
        seq_list = sorted(os.listdir(self.img_root_path))
        self.seq_list = []
        for seq_name in seq_list:
            n_frms = len(os.listdir(os.path.join(self.img_root_path, seq_name)))
            if n_frms > 0:
                self.seq_list.append(seq_name)

        if subset not in ['train', 'test']:
            raise Exception("not implement")

        self.subset = subset
        self.seq_len = seq_len
        self.load_landmark = load_landmark

        self.im_size = 40

        self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, item):
        if 'speaking' in self.seq_list[item]:
            label = 1
        elif 'silent' in self.seq_list[item]:
            label = 0
        else:
            raise NotImplementedError
        seq_path = os.path.join(self.img_root_path, self.seq_list[item])
        landmark_path = os.path.join(self.landmark_root_path, self.seq_list[item] + '.npy')
        ori_landmark = torch.from_numpy(np.load(landmark_path)).float()
        ori_landmark[ori_landmark > 39] = 39
        ori_landmark[ori_landmark < 0] = 0
        if self.subset == 'train':
            return self.train_transform(seq_path, ori_landmark, label)
        elif self.subset == 'test':
            return self.test_transform(seq_path, ori_landmark, label)

    def train_transform(self, seq_path, ori_landmark, label):
        img_names = os.listdir(seq_path)
        num_img = len(img_names)
        img_names = [os.path.join(seq_path, 'frame_{:d}.jpg'.format(id)) for id in range(num_img)]
        assert ori_landmark.shape[0] == num_img
        assert ori_landmark.shape[1] == 20

        if random.random() > 0.5:
            flip_im = True
        else:
            flip_im = False

        # print(ori_landmark.max(), ori_landmark.min())
        if label == 1 and random.random() > 0.8:
            # repeat one frames, augment silient speech
            im_id = np.random.randint(num_img)
            label = 0
            img = Image.open(img_names[im_id]).convert('RGB')
            landmark = ori_landmark[im_id]
            if flip_im:
                img = F.hflip(img)
                landmark[:,1] = 39 - landmark[:, 1]
            img = self.trans(img)
            img_seqs = []
            landmark_seqs = []
            for t in range(self.seq_len):
                img_seqs.append(img)
                landmark_seqs.append(landmark)
            img_seqs = torch.stack(img_seqs, 0) # [T, 3, H, W]
            landmark_seqs = torch.stack(landmark_seqs, 0) # [T, num_landmark, 2]
            return img_seqs, landmark_seqs, label

        if num_img < self.seq_len:
            fps = num_img / self.seq_len
            img_seqs = []
            landmark_seqs = []
            for t in range(self.seq_len):
                im_id = min(int(t * fps), num_img-1)
                # print(num_img, im_id, seq_path)
                img = Image.open(img_names[im_id]).convert('RGB')
                landmark = ori_landmark[im_id]
                if flip_im:
                    img = F.hflip(img)
                    landmark[:, 1] = 39 - landmark[:, 1]
                img = self.trans(img)
                img_seqs.append(img)
                landmark_seqs.append(landmark)
            img_seqs = torch.stack(img_seqs, 0)  # [T, 3, H, W]
            landmark_seqs = torch.stack(landmark_seqs, 0)  # [T, num_landmark, 2]
        else:
            # random sample 20 frames
            subplen = num_img // self.seq_len
            im_ids = np.array([random.randint(0, subplen - 1) for _ in range(self.seq_len)]) + np.linspace(0, num_img, self.seq_len, endpoint=False, dtype=int)
            img_seqs = []
            landmark_seqs = []
            for im_id in im_ids:
                img = Image.open(img_names[im_id]).convert('RGB')
                landmark = ori_landmark[im_id]
                if flip_im:
                    img = F.hflip(img)
                    landmark[:, 1] = 39 - landmark[:, 1]
                img = self.trans(img)
                img_seqs.append(img)
                landmark_seqs.append(landmark)
            img_seqs = torch.stack(img_seqs, 0)  # [T, 3, H, W]
            landmark_seqs = torch.stack(landmark_seqs, 0)  # [T, num_landmark, 2]


        return img_seqs, landmark_seqs, label


    def test_transform(self, seq_path, ori_landmark, label):
        img_names = os.listdir(seq_path)
        num_img = len(img_names)
        img_names = [os.path.join(seq_path, 'frame_{:d}.jpg'.format(id)) for id in range(num_img)]

        if num_img < self.seq_len:
            fps = num_img / self.seq_len
            img_seqs = []
            landmark_seqs = []
            for t in range(self.seq_len):
                im_id = min(int(t * fps), num_img-1)
                img = Image.open(img_names[im_id]).convert('RGB')
                landmark = ori_landmark[im_id]
                img = self.trans(img)
                img_seqs.append(img)
                landmark_seqs.append(landmark)
            img_seqs = torch.stack(img_seqs, 0)  # [T, 3, H, W]
            landmark_seqs = torch.stack(landmark_seqs, 0)  # [T, num_landmark, 2]
        else:
            # random sample 20 frames
            subplen = num_img // self.seq_len
            im_ids = np.array([random.randint(0, subplen - 1) for _ in range(self.seq_len)]) + np.linspace(0, num_img, self.seq_len, endpoint=False, dtype=int)
            img_seqs = []
            landmark_seqs = []
            for im_id in im_ids:
                img = Image.open(img_names[im_id]).convert('RGB')
                landmark = ori_landmark[im_id]
                img = self.trans(img)
                img_seqs.append(img)
                landmark_seqs.append(landmark)
            img_seqs = torch.stack(img_seqs, 0)  # [T, 3, H, W]
            landmark_seqs = torch.stack(landmark_seqs, 0)  # [T, num_landmark, 2]

        return img_seqs, landmark_seqs, label