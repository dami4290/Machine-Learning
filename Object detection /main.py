import pycocotools
import PIL.Image
import random
import torch
import torch.utils.data
import numpy as np
from collections import defaultdict
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import models
import torchvision.transforms as original_transforms
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
import multiprocessing as mp
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from utils import CustomBatchs, RandomHorizontalFlip, RandomVerticalFlip, Resize, show, collate_wrapper
from customdataset import NewCocoDataset
import gc

#set the Hyperparameters
n_gpus = torch.cuda.device_count()
USING_CPU = not torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()  and n_gpus > 0) else "mps")
kwargs = {'num_workers': mp.cpu_count() , 'pin_memory': True} if DEVICE.type=='cuda' else {'num_workers': mp.cpu_count()//2, 'prefetch_factor': 4}
print(f'Num of CPUs: {mp.cpu_count()}')
print(f'Device in use: {DEVICE}')
print(f'Found {n_gpus} GPU Device/s.')

#Create a dataset loader that gives a coco dataset
def main(): 
    VAL_IMG_DIR = 'coco2017/val2017'
    VAL_ANN_FILE = 'coco2017/annotations/instances_val2017.json'
    TRAIN_IMG_DIR = 'coco2017/train2017'
    TRAIN_ANN_FILE = 'coco2017/annotations/instances_train2017.json'
    USE_PRETRAINED = False
    SAVED_MODEL_PATH = '/kaggle/input/object-detection-using-pytorch/ssd300_vgg16_checkpoint_2'

    def load_dataset(transform):
        return dset.CocoDetection(root = TRAIN_IMG_DIR, 
                                annFile = TRAIN_ANN_FILE)
    
    #transformer that performs the extra datat augmentations
    transform = transforms.Compose(
    [
        transforms.RandomPhotometricDistort(),        
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),])
    
    #loading the dataset
    coco_train = load_dataset(transform=transform)
    print("Number of samples: ", len(coco_train))
    coco_train = dset.wrap_dataset_for_transforms_v2(coco_train)

    # Create a dataset loader 
    new_coco_train = NewCocoDataset(coco_train)
    data_loader = torch.utils.data.DataLoader(
        new_coco_train,
        batch_size=50 if not USING_CPU else 32,
        shuffle=True,
        collate_fn=collate_wrapper,
        **kwargs)
    
    #Load the base model
    base_model = models.get_model("ssd300_vgg16", weights=None, weights_backbone=None).train()
    
    #Initialize the weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    base_model.apply(weights_init)
    print(DEVICE)

    if (DEVICE.type == 'cuda') and (n_gpus > 1):
        base_model = nn.DataParallel(base_model, list(range(n_gpus)))
    
    #Display the loaded model
    base_model.to(DEVICE)
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    #Model Hyperparameters
    learning_rate = 1e-4
    optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    if USE_PRETRAINED:
        new_LR = 1e-5 # change this value to set a new Learning Rate for the version of notebook
    
    # if USING_CPU:
    #     checkpoint = torch.load(SAVED_MODEL_PATH, map_location=torch.device('mps'))
    # else:
    #     checkpoint = torch.load(SAVED_MODEL_PATH)
        
    # base_model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for g in optimizer.param_groups:
        g['lr'] = 1e-5
    coco_anns = pycocotools.coco.COCO(TRAIN_ANN_FILE)
    catIDs = coco_anns.getCatIds()
    cats = coco_anns.loadCats(catIDs)
    name_idx = {}
    for sub_dict in cats:
        name_idx[sub_dict["id"]] = sub_dict["name"]
    del coco_anns, catIDs, cats

    #Model Training
    EPOCHS = 5
    for epoch in range(EPOCHS):
        running_classifier_loss = 0.0
        running_bbox_loss = 0.0
        running_loss = 0.0
        counter = 0
        base_model.train()
        for data_point in tqdm(data_loader):
            _i, _t = data_point[0], data_point[1]
            if USING_CPU:
                _i = torch.stack(_i)
            _i = _i.to(DEVICE)
            _t = [{k: v.to(DEVICE) for k, v in __t.items()} for __t in _t]
            optimizer.zero_grad()
            loss_dict = base_model(_i, _t)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
            del loss_dict, losses
            counter += 1
            if counter % 500 == 499:
                last_classifier_loss = running_classifier_loss / 500 # loss per batch
                last_bbox_loss = running_bbox_loss / 500 # loss per batch
                last_loss = running_loss / 500 # loss per batch
    #             print(f'batch {counter + 1} Classification Loss: {last_classifier_loss}', end='')
    #             print(f', BBox Loss: {last_bbox_loss}')
                print(f'Epoch {epoch}, Batch {counter + 1}, Running Loss: {last_loss}')
                running_classifier_loss = 0.0
                running_bbox_loss = 0.0
                running_loss = 0.0
                
            gc.collect()
    gc.collect()

    #Load validation dataset
    def load_val_dataset(transform):
        return dset.CocoDetection(root = VAL_IMG_DIR, 
                                annFile = VAL_ANN_FILE)
    val_transform = transforms.Compose(
        [transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),])
    coco_val = load_val_dataset(transform=val_transform)
    coco_val = dset.wrap_dataset_for_transforms_v2(coco_val)

    new_coco_val = NewCocoDataset(coco_val)
    val_data_loader = torch.utils.data.DataLoader(
        new_coco_val,
        batch_size=50 if not USING_CPU else 32,
        shuffle=True,
        collate_fn=collate_wrapper,
        **kwargs
    )
    img_dtype_converter = transforms.ConvertImageDtype(torch.uint8)

    #
    data = next(iter(val_data_loader))
    _i = data[0]
    threshold = 0.5
    idx = 3
    if USING_CPU:
        _i = torch.stack(_i)
    _i = _i.to(DEVICE)
    base_model.eval()
    p_t = base_model(_i)
    confidence_length = len(np.argwhere(p_t[idx]['scores'] > threshold)[0])
    p_boxes = p_t[idx]['boxes'][: confidence_length]
    p_labels = [name_idx[i] for i in p_t[idx]['labels'][: confidence_length].tolist()]
    i_img = img_dtype_converter(_i[idx])
    annotated_image = draw_bounding_boxes(i_img, p_boxes, p_labels, colors="yellow", width=3)
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()


if __name__=='__main__':
    main()