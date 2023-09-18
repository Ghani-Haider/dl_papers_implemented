from data_setup import create_dataloaders
from engine import train
from vit_model import ViTBase
from utils import save_model, plot_curves, simple_transform
from torch import nn, optim, cuda
from torchvision import transforms
import argparse
from os import cpu_count

# training and testing directories
TRAIN_DIR = "./data/pizza_steak_sushi/train"
TEST_DIR = "./data/pizza_steak_sushi/test"

# set hyperparameter for ViT Base Model
EPOCHS = 10
BATCH_SIZE = 2
IMAGE_SIZE = 224  # image resolution
IMG_CHANNELS = 3  # image channels
NUM_CLASSES = 3 # output labels
PATCH_SIZE = 16  # dimension of the image patches
NUM_PATCH = IMAGE_SIZE**2 // PATCH_SIZE**2  # number of image patches
D_MODEL = 768  # patch embedding dimension throughout ViT
NUM_HEADS = 12 # number of heads for multiheaded attention block
NUM_LAYERS = 12 # number of encoder blocks in ViT
MLP_SIZE = 3072 # size of the MLP block in Encoder
DROPOUT_EMBEDS = 0.1 # dropout of the patch embeddings
DROPOUT_MLP = 0.1 # dropout of the MLP block in Encoder
DROPOUT_ATTN = 0 # dropout of the MSA block in Encoder

# device agnostic code
DEVICE = "cuda" if cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    # ViT image transformation
    simple_transform = transforms.Compose([
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # loading data
    train_dl, test_dl, class_names = create_dataloaders(train_dir=TRAIN_DIR,
                                                        test_dir=TEST_DIR,
                                                        transform=simple_transform,
                                                        batch_size=BATCH_SIZE,
                                                        num_workers=cpu_count())

    
    # loading model, optimizer and loss function
    vit_model = ViTBase(IMG_CHANNELS, NUM_PATCH, PATCH_SIZE, D_MODEL, NUM_HEADS,
                        NUM_LAYERS, MLP_SIZE, DROPOUT_EMBEDS, DROPOUT_MLP, DROPOUT_ATTN,
                        NUM_CLASSES).to(DEVICE)

    optimizer = optim.Adam(params=vit_model.parameters(),
                                lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), # section 4.1 (Training & Fine-tuning)
                                weight_decay=0.3)

    loss_fn = nn.CrossEntropyLoss()
    
    # train
    print(f"[INFO] training model")
    result = train(model=vit_model,
                   train_dl=train_dl,
                   test_dl=test_dl,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=DEVICE,
                   epochs=EPOCHS)
    
    # save model
    print(f"[INFO] plotting loss")
    save_model(model=vit_model, target_dir="./model", model_name="vit_landscape_classification.pth")

    # plot loss and accuracy curves
    plot_curves(result)