# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 2
image_height = 128
image_width = 128
channels = 1
save_model_dir = "saved_model/model"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
