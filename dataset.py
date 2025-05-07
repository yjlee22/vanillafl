import os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import medmnist
from PIL import Image
from medmnist.info import INFO

def get_medmnist_dataset_class(dataset_name):
    available_datasets = [
        "PathMNIST", "OCTMNIST", "PneumoniaMNIST", "ChestMNIST",
        "DermaMNIST", "RetinaMNIST", "BreastMNIST", "BloodMNIST",
        "TissueMNIST", "OrganAMNIST", "OrganCMNIST", "OrganSMNIST",
        "OrganMNIST3D", "NoduleMNIST3D", "AdrenalMNIST3D",
        "FractureMNIST3D", "VesselMNIST3D", "SynapseMNIST3D"
    ]
    for ds in available_datasets:
        if ds.lower() == dataset_name.lower():
            return ds
    raise ValueError(f"medmnist에 '{dataset_name}'에 해당하는 dataset 클래스가 없습니다. 사용 가능한 옵션은 {available_datasets} 입니다.")

class DatasetObject:
    def __init__(self, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', args=None):
        self.args = args
        self.n_client = n_client
        self.rule = rule
        self.rule_arg = rule_arg
        self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        dataset_name_lower = self.args.dataset.lower()
        self.name = f"{dataset_name_lower}_{self.n_client}_{self.seed}_{self.rule}_{rule_arg_str}"
        self.name += f"_{unbalanced_sgm:f}" if unbalanced_sgm != 0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
       
    def set_data(self):
        save_path = os.path.join(self.data_path, 'Data', self.name)
        client_x_path = os.path.join(save_path, 'client_x.npy')
        client_y_path = os.path.join(save_path, 'client_y.npy')
        test_x_path = os.path.join(save_path, 'test_x.npy')
        test_y_path = os.path.join(save_path, 'test_y.npy')

        if not (os.path.exists(client_x_path) and os.path.exists(client_y_path) and 
                os.path.exists(test_x_path) and os.path.exists(test_y_path)):
            os.makedirs(save_path, exist_ok=True)
            
            ds_class_name = self.args.dataset
            DataClass = getattr(medmnist, ds_class_name)

            info = INFO[DataClass.flag]

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

            train_dataset = DataClass(split='train', transform=transform, download=True, as_rgb=True, size=224)
            test_dataset = DataClass(split='test', transform=transform, download=True, as_rgb=True, size=224)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=len(test_dataset), shuffle=False)
            
            self.channels = info['n_channels']
            self.width = 224
            self.height = 224
            self.n_cls = len(info['label'])

            train_data = next(iter(train_loader))
            test_data = next(iter(test_loader))
            
            train_x, train_y = train_data[0].numpy(), train_data[1].numpy().reshape(-1, 1)
            test_x, test_y = test_data[0].numpy(), test_data[1].numpy().reshape(-1, 1)

            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]

            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

            n_data_per_client = int(len(train_y) / self.n_client)
            client_data_list = np.ones(self.n_client, dtype=int) * n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            if diff != 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break

            # 데이터 분할 코드
            if self.rule in ['Dirichlet', 'Pathological']:
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client, self.n_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j+1].sum() / c * 1.0

                idx_list = [np.where(train_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                
                # 각 클라이언트별로 미리 고정된 크기의 배열 생성 (리스트 형태로 유지)
                client_x = [np.zeros((client_data_list[client__], self.channels, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], 1), dtype=np.int64)
                            for client__ in range(self.n_client)]

                while np.sum(client_data_list) != 0:
                    curr_client = np.random.randint(self.n_client)
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                print("Sample distribution per class:", true_sample)
                # 리스트를 object dtype의 numpy 배열로 변환하여 저장 (각 클라이언트 데이터의 길이가 다르더라도 저장 가능)
                self.client_x = client_x
                self.client_y = client_y

            elif self.rule == 'iid':
                client_x = [np.zeros((client_data_list[client__], self.channels, self.height, self.width), dtype=np.float32)
                            for client__ in range(self.n_client)]
                client_y = [np.zeros((client_data_list[client__], 1), dtype=np.int64)
                            for client__ in range(self.n_client)]
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                self.client_x = client_x
                self.client_y = client_y

            self.test_x = test_x
            self.test_y = test_y
            
            np.save(client_x_path, np.array(self.client_x, dtype=object))
            np.save(client_y_path, np.array(self.client_y, dtype=object))
            np.save(test_x_path, test_x)
            np.save(test_y_path, test_y)
        else:
            # 저장된 client 데이터는 object array로 저장되어 있으므로 allow_pickle=True 옵션으로 로드
            self.client_x = np.load(client_x_path, allow_pickle=True)
            self.client_y = np.load(client_y_path, allow_pickle=True)
            self.n_client = len(self.client_x)
            # 테스트 데이터는 homogeneous하므로 mmap_mode 사용 가능
            self.test_x = np.load(test_x_path, mmap_mode='r')
            self.test_y = np.load(test_y_path, mmap_mode='r')
            
            ds_class_name = get_medmnist_dataset_class(self.args.dataset)
            DataClass = getattr(medmnist, ds_class_name)
            info = INFO[DataClass.flag]
            self.channels = info['n_channels']
            self.width = 224
            self.height = 224
            self.n_cls = len(info['label'])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y=True, train=False, dataset_name='', args=None):
        self.name = dataset_name
        self.args = args
       
        if self.name == self.args.dataset:
            self.train = train
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            self.X_data = data_x
            self.y_data = data_y if isinstance(data_y, bool) else data_y.astype('float32')
            ds_class_name = get_medmnist_dataset_class(self.args.dataset)
            DataClass = getattr(medmnist, ds_class_name)
            info = INFO[DataClass.flag]
            self.channels = info['n_channels']
            self.width = 224
            self.height = 224
        else:
            raise NotImplementedError
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx]
        if self.train:
            img = np.flip(img, axis=2).copy() if np.random.rand() > .5 else img
            if np.random.rand() > .5:
                pad = 4
                extended_img = np.zeros((self.channels, self.height + pad * 2, self.width + pad * 2), dtype=np.float32)
                extended_img[:, pad:-pad, pad:-pad] = img
                dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                img = extended_img[:, dim_1:dim_1+self.height, dim_2:dim_2+self.width]
        img = np.moveaxis(img, 0, -1)
        img = self.transform(img)
        return img if isinstance(self.y_data, bool) else (img, self.y_data[idx])

# DatasetFromDir 클래스는 그대로 유지
class DatasetFromDir(data.Dataset):
    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]
        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)
