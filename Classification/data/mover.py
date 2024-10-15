import shutil
import os

split_size = 0.80
cat_training_dir = "training/cat"
dog_training_dir = "training/dog"
cat_validation_dir = "validation/cat"
dog_validation_dir = "validation/dog"
training_dir=os.listdir("archive/train/train")
cat=0
dog=0
for i,img in enumerate(training_dir):
    print(img.__class__)
    if  img.startswith("cat"):
        if cat>=10000:
            img_path=os.path.join("archive/train/train",img)
            shutil.move(img_path,cat_validation_dir)
        else:
            img_path=os.path.join("archive/train/train",img)
            shutil.move(img_path,cat_training_dir)
        cat=cat+1
        
    #   shutil.move(img_path,cat_training_dir)
    else:
        if dog>=10000:
            img_path=os.path.join("archive/train/train",img)
            shutil.move(img_path,dog_validation_dir)
        else:
            img_path=os.path.join("archive/train/train",img)
            shutil.move(img_path,dog_training_dir)
        dog=dog+1
        
       

    #   shutil.move(img,cat_validation_dir)