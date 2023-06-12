import os
import zipfile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PH2Dataset(Dataset):
    def __init__(
        self,
        root,
        mode='train',
        transform=None,
        split_ratio=(0.75, 0.125, 0.125),
        seed=1,
        download=False,
        target_dir='dataset/PH2Dataset',
    ):
        self.root = root
        self.seed = seed
        self.transform = transform
        self.target_dir=target_dir
        self.download = download
        if self.download:
            self._download()
        
        assert np.isclose(np.sum(split_ratio), 1), \
            "sum of split ration should be 1"
        self.split_ratio = split_ratio
        
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode, available modes are `train`, `val`, `test`')
        
        images = []
        masks = []
        if not os.path.exists(root):
            raise RuntimeError('Provided path (`root`) does not exists')

        for root, dirs, files in os.walk(root):
            if root.endswith('_Dermoscopic_Image'):
                images.append(os.path.join(root, files[0]))
            if root.endswith('_lesion'):
                masks.append(os.path.join(root, files[0]))

        if len(images) != len(masks):
            raise RuntimeError("length mismatch, len of images differ from masks")
        
        # split dataset according to self.split_ratio
        np.random.seed(self.seed)
        ind = np.random.permutation(range(len(images)))

        split1 = int(len(images) * self.split_ratio[0])
        split2 = split1 + int(len(images) * self.split_ratio[1])
        self.train_ind = ind[:split1]
        self.val_ind = ind[split1: split2]
        self.test_ind = ind[split2:]

        assert (len(self.train_ind) + len(self.val_ind) + len(self.test_ind)) == len(images), \
                "Missed some images after splitting, error after splitting into train/val/test"

        self.mode = mode

        images = np.array(images)
        masks = np.array(masks)
        if self.mode == 'train':
            self.images = images[self.train_ind]
            self.masks = masks[self.train_ind]
        elif self.mode == 'val':
            self.images = images[self.val_ind]
            self.masks = masks[self.val_ind]
        else:
            self.images = images[self.test_ind]
            self.masks = masks[self.test_ind]

    def __len__(self):
        return len(self.images)

    def _download(self):
        if os.path.exists(self.target_dir):
            print("Found directory PH2Dataset, stop downloading!\
                \nTo download it again remove PH2Dataset directory")
            return
        
        output_file = self.target_dir + ".zip"
        download_file_from_google_drive_gdown("1uxB59F-sT-Cgqfxt0J0lMm7VZKcD1yuV", output_file, unzip=True)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
    
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        assert image.shape[1:] == mask.shape[1:], \
            f"after preprocessing images should have same shape, but you have img1: {image.shape}, img2: {mask.shape}"

        assert len(image.shape) == 3, \
            f"inccorect number of axis, you have {image.shape}, it should be (ch, w, h)"
        assert len(mask.shape) == 3, \
            f"inccorect number of axis, you have {mask.shape}, it should be (ch, w, h)"

        return image, mask


def unzip_file(folder_name, output_name):
    with zipfile.ZipFile(folder_name, 'r') as zf:
        zf.extractall(output_name)
    print("unzip done successfully you, unziped file", output_name)


def download_file_from_google_drive_gdown(id, output, unzip=True):
    # make sure you have gdown
    # if not, install it with `pip install gdown`
    try:
        import gdown
    except ImportError:
        print("Can't import gdown, probably you didn't install it, please install it with pip install gdown")
        return
    
    url = f'https://drive.google.com/uc?id={id}'
    gdown.download(url, output, quiet=False)
    print("File downloaded successfully.")
    
    if unzip and output.endswith(".zip"):
        unzip_file(output, output[:-4])


if __name__ == "__main__":
    root = 'dataset/PH2Dataset/PH2 Dataset images'
    target_size = (256, 256)
    split_ratio = (0.75, 0.125, 0.125)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),   
    ])
    train_dataset = PH2Dataset(
        root,
        mode='train',
        transform=transform,
        split_ratio=split_ratio,
        seed=1,
        download=True,
    )
    
    val_dataset = PH2Dataset(
        root,
        mode='val',
        transform=transform,
        split_ratio=split_ratio,
        seed=1,
    )
        
    test_dataset = PH2Dataset(
        root,
        mode='test',
        transform=transform,
        split_ratio=split_ratio,
        seed=1,
    )
    
    # run __getitem__
    x_train, y_train = train_dataset.__getitem__(0)
    x_val, y_val = val_dataset.__getitem__(0)
    x_test, y_test = test_dataset.__getitem__(0)
    
    def check_shapes(x, y):
        assert x.shape == (3, *target_size), f"incorrect shape of item, {x.shape}"
        assert y.shape == (1, *target_size), f"incorrect shape of item, {y.shape}"
    
    check_shapes(x_train, y_train)
    check_shapes(x_val, y_val)
    check_shapes(x_test, y_test)
    
    
    # check size
    def check_size(dataset, y):
        assert dataset.__len__() == y, f"incorrect len for train_dataset, {dataset.__len__()}"
        
    check_size(train_dataset, 150)
    check_size(val_dataset, 25)
    check_size(test_dataset, 25)
    