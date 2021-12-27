import argparse
import sys
import os
from typing import Optional

from string import ascii_lowercase

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
os.environ['TORCH_HOME'] = 'models'

import pytorch_lightning as pl

sys.path.append('utils/')
import image_utils, train_utils
sys.path.append('models/')
import ViTSTR

VOC_LIST = list(ascii_lowercase + ' ')


def create_model(freeze=False):
    model = ViTSTR.ViTSTR(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True)

    url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    #state_dict = model_zoo.load_url(url, progress=True, map_location='cpu')
    #if "model" in state_dict.keys():
    #    state_dict = state_dict["model"]
    #model.load_state_dict(state_dict, strict=False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.head = nn.Linear(192, 30)

    return model


class LitTransformer(pl.LightningModule):
    def __init__(self, freeze):
        super().__init__()
        self.model = create_model(freeze)
        self.criterion = torch.nn.BCELoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, targets = batch
        images = image_utils.reshape_image_by_patch(images).type_as(images)
        images = images.repeat(1, 3, 1, 1)

        output = self.model(images)
        loss = self.criterion(output, targets)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        images = image_utils.reshape_image_by_patch(images).type_as(images)
        images = images.repeat(1, 3, 1, 1)

        output = self.model(images)
        loss = self.criterion(output, targets)
        # Logging to TensorBoard by default
        # self.log("val_loss", loss)

        output = train_utils.tensor_to_string(output[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        target = train_utils.tensor_to_string(targets[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        score = 0
        for l in range(min(len(target), len(output))):
            if output[l] == target[l]:
                score += 1
        acc = score / len(target.rstrip())

        self.log(f"val_loss", loss, prog_bar=False)
        self.log(f"acc", acc, prog_bar=False)

        if batch_idx % 10 == 0:
            print("output : ", output, "\n")
            print("target : ", target, "\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train(path,
         dataset_max_len,
         string_len,
         batch_size,
         freeze):

    num_workers, num_gpus = (2, 1) if torch.cuda.is_available() else (0, 0)

    transformer = LitTransformer(freeze)

    trainer = pl.Trainer(max_epochs=5,
                         gpus=num_gpus
                         )
    dataset = train_utils.string_img_Dataset(img_size=(16, string_len * 2 ** 3),
                                             batch_size=batch_size,
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=path,
                                             )

    ds_len = int(len(dataset) * 0.8)

    train_set, val_set = torch.utils.data.random_split(dataset, [ds_len, len(dataset) - ds_len])

    train_set = torch.utils.data.DataLoader(train_set,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        drop_last=True)
    val_set = torch.utils.data.DataLoader(val_set,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        drop_last=True)
    trainer.fit(transformer, train_set, val_set)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="/media/tidiane/D:/Dev/CV/unsupervised_ocr/data/synth/",
                                help='Datasets path',
                                type=str)
    cmdline_parser.add_argument('-d', '--dataset_len',
                                default=6400,
                                help='dataset length',
                                type=int)
    cmdline_parser.add_argument('-l', '--str_len',
                                default=30,
                                help='string_length',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=8,
                                help='batch size',
                                type=int)
    cmdline_parser.add_argument('-f', '--freeze',
                                default=False,
                                help='freeze weights ?',
                                type=bool)

    args, unknowns = cmdline_parser.parse_known_args()

    train(args.path,
         dataset_max_len=args.dataset_len,
         string_len=args.str_len,
         batch_size=args.batch_size,
         freeze=args.freeze)
