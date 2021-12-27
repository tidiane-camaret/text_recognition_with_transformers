import argparse
import sys
import os
from typing import Optional
import pickle
from string import ascii_lowercase
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

os.environ['TORCH_HOME'] = 'models'

import pytorch_lightning as pl

sys.path.append('utils/')
import image_utils, train_utils
sys.path.append('models/')
import ViTSTR

VOC_LIST = list(ascii_lowercase + ' ')


def create_encoder(freeze=False):
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

class DiscriminatorMSE(nn.Module):
    def __init__(self, string_len, voc_len, embed_size, nb_filters):
        super(DiscriminatorMSE, self).__init__()

        self.Softmax = nn.Softmax(dim=2)
        self.Embedding = nn.Linear(in_features=voc_len, out_features=embed_size)
        # nn.Embedding(embedding_dim=1, num_embeddings=embed_size)
        self.Conv_1 = nn.Conv1d(in_channels=embed_size, out_channels=nb_filters, kernel_size=5, padding=2)
        self.LayerNorm = nn.LayerNorm(normalized_shape=[nb_filters, string_len])
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.discriminator_block = nn.Sequential(
            nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=5, padding=2),
            nn.LayerNorm(normalized_shape=[nb_filters, string_len]),
            nn.LeakyReLU(negative_slope=0.2))

        self.Conv_2 = nn.Conv1d(in_channels=nb_filters, out_channels=embed_size, kernel_size=5, padding=2)
        self.LayerNorm_2 = nn.LayerNorm(normalized_shape=[embed_size, string_len])
        self.LeakyReLU_2 = nn.LeakyReLU(negative_slope=0.2)

        self.LinearFlattened = nn.Linear(string_len*embed_size, 100)
        self.LinearFlattened2 = nn.Linear(100, 2)


        self.Linear = nn.Linear(in_features=embed_size, out_features=1)
        self.LeakyReLU_3 = nn.LeakyReLU(negative_slope=0.2)
        self.LinearFinal = nn.Linear(in_features=string_len, out_features=1)
        self.AvgPool = nn.AvgPool1d(kernel_size=string_len)
        self.Activ = nn.Tanh()

    def forward(self, x):

        x = self.Softmax(x)
        #print("softmax : ", x.shape)
        x = self.Embedding(x)
        #print("embedding : ", x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ", x.shape)
        x = self.Conv_1(x)
        #print("conv1 : ", x.shape)
        x = self.LayerNorm(x)
        #print("layernorm : ", x.shape)
        x = self.LeakyReLU(x)
        #print("ReLU : ", x.shape)

        for i in range(3):
            x = self.discriminator_block(x)
            #print(x.shape)

        x = self.Conv_2(x)
        #print("conv : ", x.shape)
        x = self.LayerNorm_2(x)
        #print(x.shape)
        x = self.LeakyReLU_2(x)

        #print("relu : ",x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print("permute : ",x.shape)
        x = self.Linear(x)
        #print("linear : ", x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ",x.shape)

        x = self.LeakyReLU_3(x)

        x = self.AvgPool(x)
        #print("avgpool : ",x.shape)
        x = torch.squeeze(x)
        #print("squeeze : ", x.shape)
        #x = self.Activ(x)

        return x


class LitTransformerGan(pl.LightningModule):
    def __init__(self, freeze, string_len, voc_len, embed_size, nb_filters, lexicon):
        super().__init__()
        self.generator = create_encoder(freeze)
        self.discriminator = DiscriminatorMSE(string_len, voc_len, embed_size, nb_filters)
        self.criterion = torch.nn.BCELoss()
        self.lexicon = lexicon
        self.string_len = string_len

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        g_opt, d_opt = self.optimizers()

        images, _ = batch
        images = image_utils.reshape_image_by_patch(images).type_as(images)
        images = images.repeat(1, 3, 1, 1)

        generated_labels = self.generator(images)

        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        batch_size = images.shape[0]

        real_label = torch.ones((batch_size), device=self.device)
        fake_label = torch.zeros((batch_size), device=self.device)

        sampled_indexes = np.random.randint(0, len(self.lexicon), batch_size)
        real_imgs = [train_utils.string_to_tensor(self.lexicon[x], self.string_len, voc_list=VOC_LIST) for x in sampled_indexes]
        example_labels = torch.stack(real_imgs).type_as(images)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.discriminator(example_labels)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.discriminator(generated_labels.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = (errD_real + errD_fake)/2

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.discriminator(generated_labels)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        images = image_utils.reshape_image_by_patch(images).type_as(images)
        images = images.repeat(1, 3, 1, 1)

        output = self.generator(images)
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
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5)
        return g_opt, d_opt

def train(images_path,
         dataset_max_len,
         string_len,
         batch_size,
         freeze,
          lexicon_path ):

    num_workers, num_gpus = (2, 1) if torch.cuda.is_available() else (0, 0)

    with open(lexicon_path , 'rb') as f:
        lexicon = pickle.load(f)

    transformer = LitTransformerGan(freeze,
                                    string_len=30,
                                    voc_len=30,
                                    embed_size=256,
                                    nb_filters=512,
                                    lexicon=lexicon)

    trainer = pl.Trainer(max_epochs=5,
                         gpus=num_gpus
                         )
    dataset = train_utils.string_img_Dataset(img_size=(16, string_len * 2 ** 3),
                                             batch_size=batch_size,
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=images_path,
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

    cmdline_parser.add_argument('-ip', '--images_path',
                                default="/home/tidiane/dev/cv/ocr/unsupervised_ocr_project/read_by_spelling_impl/data/imgs/translation_dataset/",
                                help='image dataset path',
                                type=str)
    cmdline_parser.add_argument('-lp', '--lexicon_path',
                                default="/home/tidiane/dev/cv/ocr/unsupervised_ocr_project/read_by_spelling_impl/data/lexicons/translation_dataset/exemples_strings.pkl",
                                help='lexicon path',
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

    train(images_path=args.images_path,
         dataset_max_len=args.dataset_len,
         string_len=args.str_len,
         batch_size=args.batch_size,
         freeze=args.freeze,
         lexicon_path = args.lexicon_path)
