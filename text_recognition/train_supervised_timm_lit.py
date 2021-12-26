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
    state_dict = model_zoo.load_url(url, progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
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
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images, targets = batch
        #images = images.to(device)
        #targets = targets.to(device)
        images = image_utils.reshape_image_by_patch(images)
        images = images.repeat(1, 3, 1, 1)

        output = self.model(images)
        loss = self.criterion(output, targets)
        # Logging to TensorBoard by default
        #self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        #images = images.to(device)
        #targets = targets.to(device)
        images = image_utils.reshape_image_by_patch(images)
        images = images.repeat(1, 3, 1, 1)

        output = self.model(images)
        loss = self.criterion(output, targets)
        # Logging to TensorBoard by default
        # self.log("val_loss", loss)

        output = train_utils.tensor_to_string(images[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        target = train_utils.tensor_to_string(targets[0].detach().cpu().numpy(), voc_list=VOC_LIST)
        score = 0
        for l in range(min(len(target),len(output))):
            if output[l] == target[l]:
                score += 1
        acc = score / len(target.rstrip())

        self.log(f"val_loss", loss, prog_bar=True)
        self.log(f"val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class TextDataModule(pl.LightningDataModule):
    def __init__(self,
                                             batch_size,
                                             img_size,
                                             max_len,
                                             string_tensor_length,
                                             voc_list,
                                             dataset_dir,
                                             num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_len = max_len
        self.string_tensor_length = string_tensor_length
        self.voc_list = voc_list
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers


    def setup(self, stage: Optional[str] = None):
        self.dataset = train_utils.string_img_Dataset(img_size=self.img_size,
                                             max_len=self.max_len,
                                             string_tensor_length=self.string_tensor_length,
                                             voc_list=self.voc_list,
                                             dataset_dir=self.dataset_dir,
                                             )

        ds_len = int(len(self.dataset) * 0.8)
        print("DS LEN :", ds_len)
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [ds_len, len(self.dataset) - ds_len])

        print("DS LEN :", len(self.train_set), len(self.val_set))

    def train_dataloader(self):
        train = torch.utils.data.DataLoader(self.train_set,
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          drop_last=True)

        print("TRAIN LEN :", len(train))
        return train

    def val_dataloader(self):
        val = torch.utils.data.DataLoader(self.val_set,
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          drop_last=True)
        print("VAL LEN :", len(val))

        return val
def train(path,
         dataset_max_len,
         string_len,
         batch_size,
         freeze):

    num_workers, num_gpus = (2, -1) if torch.cuda.is_available() else (0, 0)

    transformer = LitTransformer(freeze)

    trainer = pl.Trainer(max_epochs=5,
                         gpus=num_gpus
                         )
    dataset = TextDataModule(img_size=(16, string_len * 2 ** 3),
                             batch_size=batch_size,
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=path,
                                             num_workers= num_workers
                                             )
    trainer.fit(transformer, datamodule=dataset)


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
