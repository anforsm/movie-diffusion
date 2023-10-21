LOG_WANDB = True
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 40
DEVICE = "cuda"
HF_TRAIN_DATASET = "zh-plus/tiny-imagenet"
HF_TRAIN_SPLIT = "train"
HF_VAL_DATASET = "zh-plus/tiny-imagenet"
HF_VAL_SPLIT = "valid[:10%]"
HF_IMAGE_KEY = "image"
EPOCHS = 50
VAL_EVERY_N_STEPS = 50
IMAGE_EVERY_N_STEPS = 500
