import zipfile  


data_zip = "./archive.zip"
data_dir = "./data"
data_zip_ref = zipfile.ZipFile(data_zip,"r")
data_zip_ref.extractall(data_dir)

#test
test_zip = "/data/test1.zip"
test_dir = "./data"
test_zip_ref = zipfile.ZipFile(test_zip,"r")
test_zip_ref.extractall(test_dir)

#train
train_zip = "/data/train.zip"
train_dir = "./data"
train_zip_ref = zipfile.ZipFile(train_zip,"r")
train_zip_ref.extractall(train_dir)