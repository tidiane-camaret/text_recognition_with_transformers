import argparse
import sys

from string import ascii_lowercase

import torch
from vit_pytorch import ViT

sys.path.append('/home/tidiane/dev/cv/ocr/unsupervised_ocr_project/read_by_spelling_impl/')

from utils import train_utils


def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)



def train(path,
         dataset_max_len,
         string_len,
         batch_size):

    dataset = train_utils.string_img_Dataset(img_size=(32, string_len * 2 ** 5),
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=path,
                                             )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    model = ViT(
        image_size=(32, string_len * 2 ** 5),
        patch_size=32,
        num_classes=1000,
        dim=1024,
        channels=1,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )


    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(model)




    for i, (images, targets) in enumerate(train_loader):
        #print(images.shape)

        #plt.imshow(images[0].permute(1, 2, 0))
        #plt.show()

        #loss = mpp_trainer(images)

        output = model(images)
        print("output shape :", output.shape)

        #print(loss.item())
        #opt.zero_grad()
        #loss.backward()
        #opt.step()

    # save your improved network
    #torch.save(model.state_dict(), './pretrained-net.pt')

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="/media/tidiane/D:/Dev/CV/unsupervised_ocr/data/synth/",
                                help='Datasets path',
                                type=str)
    cmdline_parser.add_argument('-d', '--dataset_len',
                                default=3000,
                                help='dataset length',
                                type=int)
    cmdline_parser.add_argument('-l', '--str_len',
                                default=30,
                                help='string_length',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=30,
                                help='batch size',
                                type=int)

    args, unknowns = cmdline_parser.parse_known_args()

    train(args.path,
         dataset_max_len=args.dataset_len,
         string_len=args.str_len,
         batch_size=args.batch_size)
