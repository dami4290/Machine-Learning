import os
import glob

data_dir = ""

#create training dir
training_dir = os.path.join(data_dir,"training")
if not os.path.isdir(training_dir):
  print(training_dir)
  os.mkdir(training_dir)

#create dog in training
dog_training_dir = os.path.join(training_dir,"dog")
if not os.path.isdir(dog_training_dir):
  os.mkdir(dog_training_dir)

#create cat in training
cat_training_dir = os.path.join(training_dir,"cat")
if not os.path.isdir(cat_training_dir):
  os.mkdir(cat_training_dir)

#create validation dir
validation_dir = os.path.join(data_dir,"validation")
if not os.path.isdir(validation_dir):
  os.mkdir(validation_dir)

#create dog in validation
dog_validation_dir = os.path.join(validation_dir,"dog")
if not os.path.isdir(dog_validation_dir):
  os.mkdir(dog_validation_dir)

#create cat in validation
cat_validation_dir = os.path.join(validation_dir,"cat")
if not os.path.isdir(cat_validation_dir):
  os.mkdir(cat_validation_dir)