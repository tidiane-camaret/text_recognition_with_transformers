import argparse
import sys

from string import ascii_lowercase

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

sys.path.append('../utils/')
import image_utils
sys.path.append('../models/')
import ViTSTR


def train(path,
         dataset_max_len,
         string_len,
         batch_size,
         utils_path):

    sys.path.append(utils_path)
    from utils import train_utils

    dataset = train_utils.string_img_Dataset(img_size=(16, string_len * 2 ** 3),
                                             max_len=dataset_max_len,
                                             string_tensor_length=string_len,
                                             voc_list=ascii_lowercase + ' ',
                                             dataset_dir=path,
                                             )

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    #model = timm.create_model('vit_tiny_patch16_224',#'deit_tiny_patch16_224',
    #                          pretrained=True)
    model = ViTSTR.ViTSTR(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True)



    url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'

    state_dict = model_zoo.load_url(url, progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Linear(192, 30)

    print(model)


    opt = torch.optim.Adam(model.parameters(), lr=0.01)#3e-4)

    criterion = torch.nn.BCELoss()

    for i, (images, targets) in enumerate(train_loader):
        #plt.imshow(images[0].permute(1, 2, 0))
        #plt.show()

        images = image_utils.reshape_image_by_patch(images)

        #plt.imshow(images[0].permute(1, 2, 0))#.permute(1, 2, 0))
        #plt.show()


        #loss = mpp_trainer(images)


        #print("images shape :", images.shape)
        images = images.repeat(1, 3, 1, 1)

        output = model(images)
        #print("output shape :", output.shape)

        #features = model.forward_features(images)
        #print("features shape :", features.shape)

        #targets = train_utils.string_to_tensor(targets, 30)

        loss = criterion(output, targets)
        print(loss.item())
        print(train_utils.tensor_to_string(output[0].detach().cpu().numpy(), voc_list=list(ascii_lowercase + ' ')))
        print(train_utils.tensor_to_string(targets[0].detach().cpu().numpy(), voc_list=list(ascii_lowercase + ' ')))

        opt.zero_grad()

        loss.backward()
        opt.step()


    # save your improved network
    #torch.save(model.state_dict(), './pretrained-net.pt')

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('readbyspelling')

    cmdline_parser.add_argument('-p', '--path',
                                default="/media/tidiane/D:/Dev/CV/unsupervised_ocr/data/synth/",
                                help='Datasets path',
                                type=str)
    cmdline_parser.add_argument('-d', '--dataset_len',
                                default=64,
                                help='dataset length',
                                type=int)
    cmdline_parser.add_argument('-l', '--str_len',
                                default=30,
                                help='string_length',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=64,
                                help='batch size',
                                type=int)
    cmdline_parser.add_argument('-up', '--utils_path',
                                default='/home/tidiane/dev/cv/ocr/unsupervised_ocr_project/read_by_spelling_impl/',
                                help='utils library path',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()

    train(args.path,
         dataset_max_len=args.dataset_len,
         string_len=args.str_len,
         batch_size=args.batch_size,
         utils_path=args.utils_path)
