import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import PIL 
#Custom Batching 
class CustomBatchs:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return (self.inp, self.tgt)

def collate_wrapper(batch):
    if torch.cuda.is_available():
        return CustomBatchs(batch)
    else:
        return tuple(zip(*batch))
#Create the required data augmentations 
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.hf = transforms.RandomHorizontalFlip(1)
        
    def __call__(self, img, bboxes):
        
        if torch.rand(1)[0] < self.p:            
            img = self.hf.forward(img)
            bboxes = self.hf.forward(bboxes)
        
        return img, bboxes
    

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.vf = transforms.RandomVerticalFlip(1)
        
    def __call__(self, img, bboxes):
        if torch.rand(1)[0] < self.p:                    
            img = self.vf.forward(img)
            bboxes = self.vf.forward(bboxes)
        
        return img, bboxes

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(self.size, antialias=True)
        
    def __call__(self, img, bboxes):
        img = self.resize.forward(img)
        
        bboxes = self.resize.forward(bboxes)

        return img, bboxes
#Simple function to display the sample
def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    
    resize = Resize((300, 300))
    
    rhf = RandomHorizontalFlip()
    rvf = RandomVerticalFlip()
    image, target = sample
    
    image, bboxes = image,target["boxes"] 

    
    image, bboxes = resize(image, bboxes)
    image, bboxes = rhf(image, bboxes)
    image, bboxes = rvf(image, bboxes)
    
    if isinstance(image, PIL.Image.Image):
        image = F.to_tensor(image)
        
    image = F.convert_image_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, bboxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()