import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandomErasing

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import copy
from tqdm import tqdm
import time
import random
from scipy.linalg import expm, norm
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import json
import gc

class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):  
        data = torch.from_numpy(np.array(data, dtype=np.float32))
        return data


class PointCloudScaling(object):
    def __init__(self, 
                 scale=[2. / 3, 3. / 2], 
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 symmetries=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.from_numpy(np.array(symmetries))
        
    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        symmetries = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.symmetries = self.symmetries.to(device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1
        if hasattr(data, 'keys'):
            data['pos'] *= scale
        else:
            data *= scale
        return data


class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False, 
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'] - torch.min(data['pos'])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim+1]
                data['heights'] = height - torch.min(height)
            
            if self.centering:
                data['pos'] = data['pos'] - torch.mean(data['pos'], axis=0, keepdims=True)
            
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'] = data['pos'] / m
        else:
            if self.centering:
                data = data - torch.mean(data, axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data = data / m
        return data


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.angle):
            theta = 0
            axis = np.zeros(3)
            axis[axis_ind] = 1
            if rot_bound is not None:
                theta = np.random.uniform(-rot_bound, rot_bound)
            rot_mats.append(self.M(axis, theta))
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)

        if hasattr(data, 'keys'):
            data['pos'] = data['pos'] @ rot_mat.T
            if 'normals' in data:
                data['normals'] = data['normals'] @ rot_mat.T
        else:
            data = data @ rot_mat.T
        return data

def get_text_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data

def get_pc_data(path):
    data = {}
    dirs = os.listdir(path)
    for name in dirs:
        data[name] = {"path": os.path.join(path, name)}
        data[name]["files"] = ["points3D.txt"] # os.path.join(data[name]["path"], "points3D.txt")

    return data

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class U1652DatasetTrain2D(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            for g in gallery_imgs:
                self.pairs.append((idx, query_img, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652DatasetTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img1, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img2, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, pointcloud_img3, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # pc_imgs = []
        # for pc_path in point_cloud_paths:
        #     point_img = np.loadtxt(pc_path, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        #     if self.uniform:
        #         point_img = farthest_point_sample(point_img, self.npoints)
        #     else:
        #         point_img = point_img[0:self.npoints, :]

        #     point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        #     if not self.use_rgb:
        #         point_img = point_img[:, 0:3]
        #     pc_imgs.append(point_img)
        # point_img = np.concatenate(pc_imgs, axis=0)
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        if not self.use_rgb:
            point_img = point_img[:, 0:3]
        # np.random.shuffle(point_img)

        # point_img = PointsToTensor()(point_img)
        # point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
        # point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        # point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)

        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
            except TypeError as e:
                print(gallery_img_path)
            
        return query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652DatasetSCVGLTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 bev_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.bev_dict = get_data(bev_folder)
        self.sat_dict = get_data(gallery_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exist for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.sat_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            # sat & bev 路径
            sat_path = self.sat_dict[idx]["path"]
            sat_imgs = self.sat_dict[idx]["files"]

            bev_path = self.bev_dict[idx]["path"]
            bev_imgs = self.bev_dict[idx]["files"]

            label = self.reverse_map_dict[idx]

            for g in sat_imgs:
                if g.endswith(".gif"):
                    continue

                # 98% 使用 BEV，2% 使用 SAT
                import random
                if random.random() < 0.98 and len(bev_imgs) > 0:
                    # 随机选一个 bev
                    g_choice = "{}/{}".format(bev_path, bev_imgs[0])
                else:
                    g_choice = "{}/{}".format(sat_path, g)

                self.pairs.append((idx, label, query_img, pointcloud_img1, g_choice))
                self.pairs.append((idx, label, query_img, pointcloud_img2, g_choice))
                self.pairs.append((idx, label, query_img, pointcloud_img3, g_choice))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # pc_imgs = []
        # for pc_path in point_cloud_paths:
        #     point_img = np.loadtxt(pc_path, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        #     if self.uniform:
        #         point_img = farthest_point_sample(point_img, self.npoints)
        #     else:
        #         point_img = point_img[0:self.npoints, :]

        #     point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        #     if not self.use_rgb:
        #         point_img = point_img[:, 0:3]
        #     pc_imgs.append(point_img)
        # point_img = np.concatenate(pc_imgs, axis=0)
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
        # point_img = PointsToTensor()(point_img)
        # point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
        # point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        # point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)
        if not self.use_rgb:
            point_img = point_img[:, 0:3]
        
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
            except TypeError as e:
                print(gallery_img_path)
            
        return query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

def compute_patch_stats_from_tokens(tokens, eps=1e-6):
    """
    tokens_np: (num_patches, C) numpy array
    返回：mu (C,), sigma (C,)  -- sigma采用方差（或可用 log-variance）
    """
    tokens = tokens.to(torch.float32)
    
    # 在 dim=0 (num_patches 维度) 上计算统计量
    mu = tokens.mean(dim=1)        
    var = tokens.var(dim=1, unbiased=False)  
    sigma = torch.log(var + eps)    
    
    # concat 起来
    style_token = torch.cat([mu, sigma], dim=-1)  # (,2C)
    
    return style_token

class U1652DatasetUCVGLTrainBEV(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 bev_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 transforms_eval=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 bev_mining_model=None):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.bev_dict = get_data(bev_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exist for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        # -----------------------------
        # 1️⃣ 构建初始 pairs
        # -----------------------------
        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            bev_img = "{}/{}".format(self.bev_dict[idx]["path"],
                                       self.bev_dict[idx]["files"][0])
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # 每个pointcloud和每个gallery组合
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img1, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img2, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img3, "{}/{}".format(gallery_path, g)))

        # -----------------------------
        # 3️⃣ Dataset transforms & params
        # -----------------------------
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.transforms_eval = transforms_eval

        # -----------------------------
        # 2️⃣ 如果提供了 BEV 挖掘模型，则 Top-1 匹配 gallery
        # -----------------------------
        self.mining_confidence = {}
        if bev_mining_model is not None:
            print("\nPerforming BEV Top-1 satellite mining during init...")
            self._mine_satellite_by_bev_top1(bev_mining_model)

        torch.cuda.empty_cache()
        # 回收 Python 内存
        gc.collect()

        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, bev_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])

        if not self.use_rgb:
            point_img = point_img[:, 0:3]
        
        # Data augmentation (模仿 PointNet++ 增强)
        point_img = PointsToTensor()(point_img)
        point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
        point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)
        # 转换为 numpy 以匹配返回类型
        point_img = point_img.numpy()
        
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        bev_img = cv2.imread(bev_img_path)
        try:
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(bev_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
                bev_img = self.transforms_gallery(image=bev_img)['image']
            except TypeError as e:
                print(gallery_img_path)
                print(bev_img_path)
            
        return bev_img, query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.1)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))
    
    def _mine_satellite_by_bev_top1(self, model):
        """
        Internal function: for each BEV image, find the Top-1 most similar satellite image from gallery.
        """
        model.eval()

        bev_paths = list(set([p[3] for p in self.pairs]))
        gallery_paths = list(set([p[5] for p in self.pairs]))
        print("Unique BEVs:", len(bev_paths), "Unique Gallery:", len(gallery_paths))

        with torch.no_grad():
            bev_feats = self._extract_features(model, bev_paths)
            gallery_feats = self._extract_features(model, gallery_paths)

            # # ===== Step 1: concat for joint whitening =====
            # all_feats = torch.cat([bev_feats, gallery_feats], dim=0)

            # # ===== Step 2: Mean Centering =====
            # mean = all_feats.mean(dim=0, keepdim=True)
            # all_feats = all_feats - mean

            # # ===== Step 3: PCA Whitening =====
            # # covariance
            # cov = torch.mm(all_feats.T, all_feats) / (all_feats.size(0) - 1)

            # # eigen decomposition
            # eigvals, eigvecs = torch.linalg.eigh(cov)

            # # sort descending
            # idx = torch.argsort(eigvals, descending=True)
            # eigvals = eigvals[idx]
            # eigvecs = eigvecs[:, idx]

            # # whitening transform
            # eps = 1e-6
            # whitening_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T

            # all_feats = all_feats @ whitening_matrix

            # # ===== Step 4: split back =====
            # bev_feats = all_feats[:len(bev_feats)]
            # gallery_feats = all_feats[len(bev_feats):]

            # # ===== Step 5: L2 normalize =====
            # bev_feats = F.normalize(bev_feats, dim=1)
            # gallery_feats = F.normalize(gallery_feats, dim=1)

        sim = bev_feats @ gallery_feats.T  # (Nb, Ns)
        best_idx = sim.argmax(dim=1)

        correct = 0
        total = len(bev_paths)
        
        for i in range(total):
            bev_path = bev_paths[i]
            matched_gallery_path = gallery_paths[best_idx[i]]
            
            # 提取路径中的最后 4 位数字 (假设文件名格式为 ...XXXX.png)
            bev_id = self._get_file_id(bev_path)
            gallery_id = self._get_file_id(matched_gallery_path)
            
            if bev_id == gallery_id:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0.0
        print(f"BEV -> Satellite Top-1 Accuracy: {accuracy:.4f} ({correct}/{total})")
        # --- [新增] 计算匹配准确率 End ---

        # 4. 建立 BEV -> Top-1 gallery 映射
        bev_to_gallery = {bev_paths[i]: gallery_paths[best_idx[i]] for i in range(len(best_idx))}

        # 5. 重建 pairs
        new_pairs = []
        for p in self.pairs:
            idx, label, query_img, bev_img, pc_path, _ = p
            best_gallery = bev_to_gallery[bev_img]
            new_pairs.append((idx, label, query_img, bev_img, pc_path, best_gallery))

        self.pairs = new_pairs
        self.samples = copy.deepcopy(self.pairs)
        print("BEV Top-1 satellite mining done during init.")

    def _extract_features(self, model, img_paths, device='cuda'):
        style_tokens = []
        # feats = []
        for path in tqdm(img_paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transforms_eval is not None:
                img = self.transforms_eval(image=img)['image']
            img = img.unsqueeze(0).float().to(device)
            with torch.no_grad():
                _, _, style_token = model(img)
            style_tokens.append(style_token)

        style_tokens = torch.cat(style_tokens, dim=0)
        # style_tokens = compute_patch_stats_from_tokens(style_tokens)
        return style_tokens
    
    def _get_file_id(self, path):
        """
        Helper: Extract the last 4 digits from the filename.
        Example: '/data/scene_1234/bev_5678.png' -> '5678'
        """
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)  # 去掉后缀 .png/.jpg
        # 取文件名的最后 4 个字符作为 ID
        return name[-4:]

class U1652DatasetUCVGLTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 bev_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 transforms_bev=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.bev_dict = get_data(bev_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]

            bev_img = "{}/{}".format(self.bev_dict[idx]["path"],
                                       self.bev_dict[idx]["files"][0])
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img1, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img2, "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, bev_img, pointcloud_img3, "{}/{}".format(gallery_path, g)))

        # For UCVGL
        galleries = [p[-1] for p in self.pairs]   # 提取最后一列
        random.shuffle(galleries)                 # 打乱 gallery

        # 重新组合
        self.pairs = [(p[0], p[1], p[2], p[3], p[4], g) for p, g in zip(self.pairs, galleries)]

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.transforms_bev = transforms_bev
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, bev_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        # point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])

        if not self.use_rgb:
            point_img = point_img[:, 0:3]

        # Data augmentation (模仿 PointNet++ 增强) - 仅在训练时应用
        point_img = PointsToTensor()(point_img)
        point_img = PointCloudScaling(scale=[0.95, 1.05])(point_img)  # 减小缩放范围
        point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        point_img = PointCloudRotation(angle=[0.0, 0.5, 0.0])(point_img)  # 减小旋转角度
        point_img = point_img.numpy()

        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        bev_img = cv2.imread(bev_img_path)
        try:
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(bev_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None and self.transforms_bev is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
                bev_img = self.transforms_bev(image=bev_img)['image']
            except TypeError as e:
                print(gallery_img_path)
                print(bev_img_path)
            
        return bev_img, query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652DatasetUCVGLTrainSampled(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 bev_folder,
                 pointcloud_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 transforms_bev=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 gallery_ratio=0.1):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        self.bev_dict = get_data(bev_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        # Collect all gallery paths
        all_gallery_paths = []
        for idx in self.ids:
            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    all_gallery_paths.append("{}/{}".format(gallery_path, g))

        # Randomly select a subset of gallery based on ratio
        num_gallery = int(len(all_gallery_paths) * gallery_ratio)
        self.selected_gallery = random.sample(all_gallery_paths, num_gallery)

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])
            
            pointcloud_img1 = "{}/{}".format(self.pointcloud_dict[idx+'_group0']["path"],
                                            self.pointcloud_dict[idx+'_group0']["files"][0])
            pointcloud_img2 = "{}/{}".format(self.pointcloud_dict[idx+'_group1']["path"],
                                            self.pointcloud_dict[idx+'_group1']["files"][0])
            pointcloud_img3 = "{}/{}".format(self.pointcloud_dict[idx+'_group2']["path"],
                                            self.pointcloud_dict[idx+'_group2']["files"][0])

            bev_img = "{}/{}".format(self.bev_dict[idx]["path"],
                                       self.bev_dict[idx]["files"][0])
            
            label = self.reverse_map_dict[idx]

            for g_path in self.selected_gallery:
                self.pairs.append((idx, label, query_img, bev_img, pointcloud_img1, g_path))
                self.pairs.append((idx, label, query_img, bev_img, pointcloud_img2, g_path))
                self.pairs.append((idx, label, query_img, bev_img, pointcloud_img3, g_path))

        # For UCVGL - shuffle galleries
        galleries = [p[-1] for p in self.pairs]   # 提取最后一列
        random.shuffle(galleries)                 # 打乱 gallery

        # 重新组合
        self.pairs = [(p[0], p[1], p[2], p[3], p[4], g) for p, g in zip(self.pairs, galleries)]

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.transforms_bev = transforms_bev
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, bev_img_path, point_cloud_paths, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
        if self.uniform:
            point_img = farthest_point_sample(point_img, self.npoints)
        else:
            point_img = point_img[0:self.npoints, :]

        # point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])

        if not self.use_rgb:
            point_img = point_img[:, 0:3]

        # Data augmentation (模仿 PointNet++ 增强) - 仅在训练时应用
        point_img = PointsToTensor()(point_img)
        point_img = PointCloudScaling(scale=[0.95, 1.05])(point_img)  # 减小缩放范围
        point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
        point_img = PointCloudRotation(angle=[0.0, 0.5, 0.0])(point_img)  # 减小旋转角度
        point_img = point_img.numpy()

        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        bev_img = cv2.imread(bev_img_path)
        try:
            bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(bev_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            bev_img = cv2.flip(bev_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None and self.transforms_bev is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
                bev_img = self.transforms_bev(image=bev_img)['image']
            except TypeError as e:
                print(gallery_img_path)
                print(bev_img_path)
            
        return bev_img, query_img, gallery_img, point_img.astype(np.float32), idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))


class U1652_DatasetVLN2DTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 pointcloud_uniform=True,
                 use_rgb=False,
                 pointcloud_npoints=1024,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)
        
        self.uniform = pointcloud_uniform
        self.npoints = pointcloud_npoints
        self.use_rgb = use_rgb

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.query_dict[idx]["path"],
                                       self.query_dict[idx]["files"][0])

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_imgs = self.gallery_dict[idx]["files"]
            
            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                if g[:-4] != '.gif':
                    # self.pairs.append((idx, label, query_img, (pointcloud_img1, pointcloud_img2, pointcloud_img3), "{}/{}".format(gallery_path, g)))
                    self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        gallery_img = cv2.imread(gallery_img_path)
        try:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(gallery_img_path)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            try:
                gallery_img = self.transforms_gallery(image=gallery_img)['image']
            except TypeError as e:
                print(gallery_img_path)
            
        return query_img, gallery_img, idx, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, ):

        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.2)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

class U1652DatasetEvalVLN2D(Dataset):

    def __init__(self,
                 json_file,
                 data_folder,
                 mode = 'query',
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()
        self.mode = mode
        self.instruction_dict = get_text_data(json_file)
        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.texts = []
        self.sample_ids = []

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                    file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        sample_id = self.sample_ids[index]
        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1


        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

class U1652DatasetEval_(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label, img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)
    
class U1652DatasetEval3d(Dataset):

    def __init__(self,
                 data_folder,
                 pointcloud_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 pointcloud_uniform=True,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)
        self.pointcloud_dict = get_pc_data(pointcloud_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms
        self.uniform = pointcloud_uniform

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        if mode == 'query':
            self.point_clouds = []
            self.pointcloud_dict = get_pc_data(pointcloud_folder)

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))
                self.sample_ids.append(sample_id)
                if mode == 'query':
                    self.point_clouds.append("{}/{}".format(self.pointcloud_dict[sample_id]["path"],
                                                  self.pointcloud_dict[sample_id]["files"][0]))

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == 'query':
            point_cloud_paths = self.point_clouds[index]
            point_img = np.loadtxt(point_cloud_paths, delimiter=' ', skiprows=3, usecols=range(1, 7))    # x, y, z, r, g, b
            if self.uniform:
                point_img = farthest_point_sample(point_img, self.npoints)
            else:
                point_img = point_img[0:self.npoints, :]

            point_img[:, 0:3] = pc_normalize(point_img[:, 0:3])
            # point_img = PointsToTensor()(point_img)
            # point_img = PointCloudScaling(scale=[0.9, 1.1])(point_img)
            # point_img = PointCloudCenterAndNormalize(gravity_dim=1)(point_img)
            # point_img = PointCloudRotation(angle=[0.0, 1.0, 0.0])(point_img)
            if not self.use_rgb:
                point_img = point_img[:, 0:3]

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, point_img.astype(np.float32), label, img_path

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

class U1652DatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))
                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)

    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   drone_color_jitter_strength=0.15):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                    #   A.OneOf([
                                    #     A.ToSepia(p=0.5),
                                    #     A.Solarize(threshold=128, p=0.5),
                                    # ], p=0.1),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    # train_bev_transforms = A.Compose([
    #     # 1. 几何：随机裁剪 + 轻微透视/仿射 (模拟重建误差)
    #     A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.75, 1.0), p=1.0),
    #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    #     A.Perspective(scale=(0.01, 0.05), p=0.3),
        
    #     # 2. 颜色：【关键】强制降低亮度，模拟卫星图的光照环境
    #     # 注意：brightness 下限设为负数，强制变暗
    #     A.ColorJitter(brightness=(0.4, 0.9), contrast=(0.7, 1.3),  # 【降低下限】
    #              saturation=(0.7, 1.2), p=0.6),
    #     # 3. 直方图均衡化：【强烈推荐】消除整体亮度差异，只保留结构
    #     A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    #     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
    #     # 【增强】点云特性 - 更强调稀疏性和噪点
    #     A.OneOf([
    #         A.GaussNoise(var_limit=(30.0, 80.0), p=1.0),  # 【增强】
    #         A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
    #         A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0),  # 【新增】模拟点云缺失
    #     ], p=0.6),  # 【提高概率】
        
    #     # 边缘增强
    #     A.OneOf([
    #         A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    #         A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
    #     ], p=0.3),
        
    #     # Dropout - 模拟稀疏性
    #     A.GridDropout(ratio=0.35, unit_size_min=5, unit_size_max=15, p=0.6),  # 【提高概率】
        
    #     A.Normalize(mean, std),
    #     ToTensorV2(),
    # ])
    
    # train_sat_transforms = A.Compose([
    #     # 1. 几何：随机裁剪 + 【关键】透视变换 (模拟非正射视角)
    #     A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.75, 1.0), p=1.0),
    #     A.Perspective(scale=(0.02, 0.08), p=0.5), # 模拟UAV视角的畸变
    #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        
    #     # 【修改】颜色 - Satellite较暗，主要变亮
    #     A.ColorJitter(brightness=(1.0, 1.5), contrast=(0.8, 1.3),  # 【提高上限】
    #                 saturation=(0.8, 1.3), p=0.6),
    #     A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    #     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        
    #     # 纹理破坏 - 模拟BEV的稀疏
    #     A.OneOf([
    #         A.GaussianBlur(blur_limit=(3, 9), p=1.0),
    #         A.MotionBlur(blur_limit=5, p=1.0),
    #     ], p=0.5),
        
    #     # 边缘增强
    #     A.OneOf([
    #         A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    #         A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
    #     ], p=0.3),
        
    #     # Dropout - 与BEV对称
    #     A.GridDropout(ratio=0.3, unit_size_min=5, unit_size_max=15, p=0.5),
        
    #     # 压缩伪影
    #     A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),  # 【降低下限】
        
    #     A.Normalize(mean, std),
    #     ToTensorV2(),
    # ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        # A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=1.0),
                                        A.ColorJitter(brightness=drone_color_jitter_strength,
                                                      contrast=drone_color_jitter_strength,
                                                      saturation=drone_color_jitter_strength,
                                                      hue=drone_color_jitter_strength,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                    #     A.OneOf([
                                    #     A.ToSepia(p=0.5),
                                    #     A.Solarize(threshold=128, p=0.5),
                                    # ], p=0.1),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                            # A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0),  # 【新增】模拟点云缺失
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms

def get_transforms1(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    # train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
    #                                   A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
    #                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
    #                                                 always_apply=False, p=0.5),
    #                                   A.OneOf([
    #                                       A.AdvancedBlur(p=1.0),
    #                                       A.Sharpen(p=1.0),
    #                                   ], p=0.3),
    #                                   A.OneOf([
    #                                       A.GridDropout(ratio=0.4, p=1.0),
    #                                       A.CoarseDropout(max_holes=25,
    #                                                       max_height=int(0.2 * img_size[0]),
    #                                                       max_width=int(0.2 * img_size[0]),
    #                                                       min_holes=10,
    #                                                       min_height=int(0.1 * img_size[0]),
    #                                                       min_width=int(0.1 * img_size[0]),
    #                                                       p=1.0),
    #                                   ], p=0.3),
    #                                 #   A.OneOf([
    #                                 #     A.ToSepia(p=0.5),
    #                                 #     A.Solarize(threshold=128, p=0.5),
    #                                 # ], p=0.1),
    #                                   A.RandomRotate90(p=1.0),
    #                                   A.Normalize(mean, std),
    #                                   ToTensorV2(),
    #                                   ])

    train_bev_transforms = A.Compose([
        # 1. 几何：随机裁剪 + 轻微透视/仿射 (模拟重建误差)
        A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # 2. 颜色：【关键】强制降低亮度，模拟卫星图的光照环境
        # 注意：brightness 下限设为负数，强制变暗
        A.ColorJitter(brightness=(0.5, 0.9), contrast=(0.8, 1.2), saturation=(0.8, 1.2), p=0.6),
        
        # 3. 直方图均衡化：【强烈推荐】消除整体亮度差异，只保留结构
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        # 4. 噪声：模拟点云重建的噪点
        A.OneOf([
            A.GaussNoise(var_limit=(20.0, 60.0), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.4),
        
        A.Normalize(mean, std),
        ToTensorV2(),
    ])
    
    train_sat_transforms = A.Compose([
        # 1. 几何：随机裁剪 + 【关键】透视变换 (模拟非正射视角)
        A.RandomResizedCrop(img_size[0], img_size[1], scale=(0.8, 1.0), p=1.0),
        A.Perspective(scale=(0.02, 0.08), p=0.5), # 模拟UAV视角的畸变
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # 2. 颜色：【关键】强制提高亮度，模拟BEV的高亮环境
        # 注意：brightness 下限设为正数或0，鼓励变亮
        A.ColorJitter(brightness=(1.0, 1.4), contrast=(0.8, 1.2), saturation=(0.8, 1.2), p=0.6),
        
        # 3. 直方图均衡化：与BEV保持一致的处理逻辑
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        
        # 4. 纹理破坏：【关键】模拟点云的稀疏和缺乏纹理
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0), # 模糊掉高频纹理
            A.GridDropout(ratio=0.3, unit_size_min=5, unit_size_max=15, p=1.0), # 模拟点云空洞
            A.CoarseDropout(max_holes=10, max_height=20, max_width=20, p=1.0),
        ], p=0.5),
        
        # 5. 压缩伪影：模拟传输或重建过程中的质量损失
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
        
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                    #     A.OneOf([
                                    #     A.ToSepia(p=0.5),
                                    #     A.Solarize(threshold=128, p=0.5),
                                    # ], p=0.1),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms2(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation (BEV & Satellite)
    # -------------------------------------------------
    geom_aug = [
        # 更大的尺度变化（CVGL非常重要）
        A.RandomResizedCrop(img_size[0], img_size[1],
                            scale=(0.5, 1.0),
                            ratio=(0.75, 1.33),
                            p=1.0),

        # 方向不变性（CVGL关键）
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

        # 小幅仿射
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3),
    ]

    # -------------------------------------------------
    # BEV transform
    # -------------------------------------------------
    train_bev_transforms = A.Compose(

        geom_aug +

        [
            # BEV亮度通常偏高
            A.ColorJitter(
                brightness=(0.6, 1.0),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.6),

            # 减少颜色依赖
            A.ToGray(p=0.2),

            # 直方图均衡
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

            # 模拟点云重建噪声
            A.OneOf([
                A.GaussNoise(var_limit=(20, 60)),
                A.ISONoise(),
            ], p=0.4),

            # 模拟BEV空洞
            A.OneOf([
                A.GridDropout(ratio=0.4),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=25,
                    max_width=25)
            ], p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # -------------------------------------------------
    train_sat_transforms = A.Compose(

        geom_aug +

        [
            # UAV视角模拟
            A.Perspective(scale=(0.02, 0.08), p=0.5),

            # 卫星亮度偏暗
            A.ColorJitter(
                brightness=(0.8, 1.3),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.6),

            # 纹理削弱
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.Downscale(scale_min=0.6, scale_max=0.9),
            ], p=0.4),

            # 结构增强
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            # 模拟缺失区域
            A.CoarseDropout(
                max_holes=10,
                max_height=20,
                max_width=20,
                p=0.3),

            # 压缩伪影
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------
    train_drone_transforms = A.Compose([

        A.RandomResizedCrop(img_size[0], img_size[1],
                            scale=(0.6, 1.0),
                            ratio=(0.75, 1.33),
                            p=1.0),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.6),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.6),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.AdvancedBlur(),
        ], p=0.3),

        A.OneOf([
            A.GridDropout(ratio=0.4),
            A.CoarseDropout(
                max_holes=25,
                max_height=int(0.2 * img_size[0]),
                max_width=int(0.2 * img_size[0]),
                min_holes=10,
                min_height=int(0.1 * img_size[0]),
                min_width=int(0.1 * img_size[0]),
            ),
        ], p=0.3),

        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms3(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation (BEV & Satellite)
    # 关键：缩小scale范围，保持几何一致性
    # -------------------------------------------------
    geom_aug = [
        # 缩小scale范围：从(0.5, 1.0)改为(0.75, 1.0)
        # 避免过度缩放导致BEV和Satellite失去对应关系
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.75, 1.0),        # 关键修改：缩小范围
            ratio=(0.9, 1.1),         # 关键修改：接近正方形
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        # 旋转保持180度（鸟瞰图方向不变性）
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.8),

        # 小幅仿射变换（仅平移和微小缩放）
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,         # 降低scale_limit
            rotate_limit=5,           # 小角度旋转
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
    ]

    # -------------------------------------------------
    # BEV transform
    # 重点：模拟点云特性，而非几何变形
    # -------------------------------------------------
    train_bev_transforms = A.Compose(
        geom_aug +
        [
            # BEV点云噪声模拟
            A.OneOf([
                A.GaussNoise(var_limit=(30, 80)),    # 增加噪声强度
                A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.3, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.5),

            # 模拟点云稀疏性和空洞（关键！）
            A.OneOf([
                A.GridDropout(
                    ratio=0.3,              # 适度dropout
                    unit_size_min=8,
                    unit_size_max=30,
                    holes_number_x=8,
                    holes_number_y=8,
                    random_offset=True,
                    fill_value=0,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=20,
                    max_width=20,
                    min_holes=10,
                    fill_value=0,
                ),
            ], p=0.5),

            # BEV亮度/对比度调整
            A.ColorJitter(
                brightness=(0.7, 1.1),      # 缩小范围
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.5
            ),

            # 降低颜色依赖（点云通常无颜色或颜色不可靠）
            A.ToGray(p=0.3),

            # 模拟点云边界模糊
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ], p=0.4),

            # 直方图均衡
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # 重点：模拟卫星图像特性，保持几何与BEV一致
    # -------------------------------------------------
    train_sat_transforms = A.Compose(
        geom_aug +  # 使用相同的geom_aug，确保几何一致性！
        [
            # 卫星图像光照变化
            A.ColorJitter(
                brightness=(0.8, 1.2),      # 缩小范围
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
                p=0.6
            ),

            # 模拟季节性变化
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            # 卫星图像模糊/分辨率变化
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_AREA),
            ], p=0.4),

            # 轻微透视（模拟卫星倾斜拍摄）
            A.Perspective(scale=(0.02, 0.05), p=0.3),

            # 结构增强
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            # 模拟遮挡（树木、云层等）
            A.CoarseDropout(
                max_holes=8,
                max_height=15,
                max_width=15,
                min_holes=4,
                fill_value=0,
                p=0.3
            ),

            # 压缩伪影
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------
    train_drone_transforms = A.Compose([
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),           # 缩小scale范围
            ratio=(0.9, 1.1),           # 接近正方形
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
            p=0.5
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=15,
            max_height=15,
            max_width=15,
            min_holes=8,
            p=0.3
        ),

        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms5(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation
    # -------------------------------------------------
    geom_aug = [
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.8),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
    ]

    # -------------------------------------------------
    # BEV transform
    # -------------------------------------------------
    train_bev_transforms = A.Compose(
        geom_aug +
        [
            # 新增：模拟点云重建局部形变（安全版）
            A.ElasticTransform(
                alpha=1,
                sigma=10,
                alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),

            A.OneOf([
                A.GaussNoise(var_limit=(30, 80)),
                A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.3, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.5),

            A.OneOf([
                A.GridDropout(
                    ratio=0.3,
                    unit_size_min=8,
                    unit_size_max=30,
                    holes_number_x=8,
                    holes_number_y=8,
                    random_offset=True,
                    fill_value=0,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=20,
                    max_width=20,
                    min_holes=10,
                    fill_value=0,
                ),
            ], p=0.5),

            A.ColorJitter(
                brightness=(0.7, 1.1),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.5
            ),

            # 修改：降低灰度概率
            A.ToGray(p=0.15),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ], p=0.4),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # -------------------------------------------------
    train_sat_transforms = A.Compose(
        geom_aug +
        [
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
                p=0.6
            ),

            # 新增：真实光照变化
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # 新增：阴影模拟
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_AREA),
            ], p=0.4),

            A.Perspective(scale=(0.02, 0.05), p=0.3),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.CoarseDropout(
                max_holes=8,
                max_height=15,
                max_width=15,
                min_holes=4,
                fill_value=0,
                p=0.3
            ),

            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------
    train_drone_transforms = A.Compose([
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

        A.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=0.05,
            p=0.5
        ),

        # 新增：gamma增强
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=15,
            max_height=15,
            max_width=15,
            min_holes=8,
            p=0.3
        ),

        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms6(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation
    # -------------------------------------------------
    geom_aug = [
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.8),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
    ]

    # -------------------------------------------------
    # BEV transform
    # -------------------------------------------------
    train_bev_transforms = A.Compose(
        geom_aug +
        [
            # 新增：模拟点云重建局部形变（安全版）
            A.ElasticTransform(
                alpha=0.5,
                sigma=10,
                alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.1
            ),

            A.OneOf([
                A.GaussNoise(var_limit=(30, 80)),
                A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.3, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.5),

            A.OneOf([
                A.GridDropout(
                    ratio=0.25,
                    unit_size_min=8,
                    unit_size_max=30,
                    holes_number_x=8,
                    holes_number_y=8,
                    random_offset=True,
                    fill_value=0,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=20,
                    max_width=20,
                    min_holes=10,
                    fill_value=0,
                ),
            ], p=0.5),

            A.ColorJitter(
                brightness=(0.7, 1.1),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.5
            ),

            # 修改：降低灰度概率
            A.ToGray(p=0.3),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ], p=0.4),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # -------------------------------------------------
    train_sat_transforms = A.Compose(
        geom_aug +
        [
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
                p=0.6
            ),

            # 新增：真实光照变化
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # 新增：阴影模拟
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_AREA),
            ], p=0.4),

            A.Perspective(scale=(0.01, 0.03), p=0.2),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.CoarseDropout(
                max_holes=8,
                max_height=15,
                max_width=15,
                min_holes=4,
                fill_value=0,
                p=0.3
            ),

            A.Sharpen(
                alpha=(0.1,0.2),
                lightness=(0.9,1.1),
                p=0.3
            ),

            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------
    train_drone_transforms = A.Compose([
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

        A.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=0.05,
            p=0.5
        ),

        # 新增：gamma增强
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=15,
            max_height=15,
            max_width=15,
            min_holes=8,
            p=0.3
        ),

        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.4),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms7(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation
    # -------------------------------------------------
    geom_aug = [
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.8),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
    ]

    # -------------------------------------------------
    # BEV transform
    # -------------------------------------------------
    train_bev_transforms = A.Compose(
        geom_aug +
        [
            # 新增：模拟点云重建局部形变（安全版）
            A.ElasticTransform(
                alpha=1,
                sigma=10,
                alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),

            A.OneOf([
                A.GaussNoise(var_limit=(30, 80)),
                A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.3, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.5),

            A.OneOf([
                A.GridDropout(
                    ratio=0.3,
                    unit_size_min=8,
                    unit_size_max=30,
                    holes_number_x=8,
                    holes_number_y=8,
                    random_offset=True,
                    fill_value=0,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=20,
                    max_width=20,
                    min_holes=10,
                    fill_value=0,
                ),
            ], p=0.5),

            A.ColorJitter(
                brightness=(0.7, 1.1),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.5
            ),

            A.ToGray(p=0.15),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ], p=0.4),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # -------------------------------------------------
    train_sat_transforms = A.Compose(
        geom_aug +
        [
            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
                p=0.6
            ),

            # 新增：真实光照变化
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # 新增：阴影模拟
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_AREA),
            ], p=0.4),

            A.Perspective(scale=(0.02, 0.05), p=0.3),

            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),

            A.CoarseDropout(
                max_holes=8,
                max_height=15,
                max_width=15,
                min_holes=4,
                fill_value=0,
                p=0.3
            ),

            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.4),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------
    train_drone_transforms = A.Compose([
        A.RandomResizedCrop(
            img_size[0], img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),

        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

        A.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=0.05,
            p=0.5
        ),

        # 新增：gamma增强
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=15,
            max_height=15,
            max_width=15,
            min_holes=8,
            p=0.3
        ),

        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.4),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms

def get_transforms8(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    # -------------------------------------------------
    # Shared geometry augmentation
    # -------------------------------------------------

    geom_aug = [

        A.RandomResizedCrop(
            img_size[0],
            img_size[1],
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR_EXACT,
            p=1.0
        ),

        A.Rotate(
            limit=180,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4
        ),
    ]

    # -------------------------------------------------
    # BEV transform
    # -------------------------------------------------

    train_bev_transforms = A.Compose(
        geom_aug + [

            A.ElasticTransform(
                alpha=0.5,
                sigma=10,
                alpha_affine=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.1
            ),

            A.OneOf([
                A.GaussNoise(var_limit=(30, 80)),
                A.ISONoise(color_shift=(0.05, 0.15), intensity=(0.3, 0.6)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            ], p=0.5),

            A.OneOf([
                A.GridDropout(
                    ratio=0.25,
                    unit_size_min=8,
                    unit_size_max=30,
                    holes_number_x=8,
                    holes_number_y=8,
                    random_offset=True,
                    fill_value=0,
                ),
                A.CoarseDropout(
                    max_holes=20,
                    max_height=20,
                    max_width=20,
                    min_holes=10,
                    fill_value=0,
                ),
            ], p=0.5),

            A.ColorJitter(
                brightness=(0.7, 1.1),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                p=0.5
            ),

            # 新增：阴影模拟
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),

            A.ToGray(p=0.15),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.4),

            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.4
            ),

            A.ImageCompression(
                quality_lower=85,
                quality_upper=100,
                p=0.4
            ),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # Satellite transform
    # -------------------------------------------------

    train_sat_transforms = A.Compose(
        geom_aug + [

            A.ColorJitter(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
                p=0.6
            ),

            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # 新增：阴影模拟
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MedianBlur(blur_limit=3),
                A.Downscale(scale_min=0.7, scale_max=0.9),
            ], p=0.4),

            A.Perspective(
                scale=(0.01, 0.03),
                p=0.2
            ),

            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.4
            ),

            A.CoarseDropout(
                max_holes=8,
                max_height=15,
                max_width=15,
                min_holes=4,
                fill_value=0,
                p=0.3
            ),

            A.Sharpen(
                alpha=(0.1, 0.2),
                lightness=(0.9, 1.1),
                p=0.3
            ),

            A.ImageCompression(
                quality_lower=85,
                quality_upper=100,
                p=0.4
            ),

            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    # -------------------------------------------------
    # UAV transform
    # -------------------------------------------------

    train_drone_transforms = A.Compose([

        A.RandomResizedCrop(
            img_size[0],
            img_size[1],
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            p=1.0
        ),

        A.Rotate(
            limit=180,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ),

        A.ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=0.05,
            p=0.5
        ),

        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=15,
            max_height=15,
            max_width=15,
            min_holes=8,
            p=0.3
        ),

        A.ImageCompression(
            quality_lower=85,
            quality_upper=100,
            p=0.4
        ),

        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms, train_bev_transforms
