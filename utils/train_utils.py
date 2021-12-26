from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
import pickle
import random
import torch
from string import ascii_letters
import numpy as np

class string_img_Dataset(Dataset):

    def __init__(self, img_size, max_len, voc_list=ascii_letters + ' ', string_tensor_length=10, dataset_dir="data/synth/", batch_size=None):
        with open(dataset_dir+'img_labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)

        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.string_tensor_length = string_tensor_length
        self.max_len = max_len
        self.voc_list = voc_list
        self.transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.1307,), (0.3081,))
                        transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return min(len(self.labels), self.max_len)
    def __getitem__(self, idx):

        label = self.labels[idx]

        img_idx = label[0][1]
        img = Image.open(self.dataset_dir+"img"+str(img_idx)+".png")

        img_tensor = self.transform(img)

        string = [l[0] for l in label]


        string_tensor = string_to_tensor(string, self.string_tensor_length, voc_list=self.voc_list)
        #print(string_tensor.size())
        return img_tensor, string_tensor

    def get_train_indices(self):
        sel_length = np.random.choice(self.lexicon_lengths)
        all_indices = np.where([self.lexicon_lengths[i] == sel_length for i in np.arange(len(self.lexicon_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices


def get_rand_strings_from_lexicon(string_len, batch_size: int = 64, lexfilename ='/home/tidiane/dev/cv/ocr/unsupervised_ocr_project/readbyspelling_impl/data/brown_ds/brown_rand_strings.pkl'):
    with open(lexfilename, 'rb') as f:
        lexicon = pickle.load(f)

    # Sample indexes in the lexicon
    sampled_indexes = np.random.randint(0, len(lexicon), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1,0] * batch_size

    # Generate a list of binary numbers for training.
    data = [lexicon[x] for x in sampled_indexes]
    #padding ( à revoir )data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data

def get_rand_strings(string_len, batch_size: int = 64):
    data = []
    for i in range(batch_size):
        s = ''.join(random.choices(ascii_letters + ' ', k=string_len))
        data.append(s)


    return data


def string_to_tensor(string, tensor_length, sos=False, voc_list=ascii_letters + ' ', nb_spe_char=3):

    voc_size = len(voc_list)
    sos_index = voc_size
    pad_index = voc_size + 1
    unk_index = voc_size + 2
    tensor = torch.zeros(tensor_length, voc_size + nb_spe_char)
    if sos:
        tensor[0][sos_index] = 1  # initialisation with token 0

    for idx in range(0, tensor_length):
        if idx < len(string):
            if string[idx] in voc_list:
                letter_idx = voc_list.index(string[idx])
            else:
                letter_idx = unk_index
        else:
            letter_idx = pad_index
        if sos:
            tensor[idx+1][letter_idx] = 1
        else:
            tensor[idx][letter_idx] = 1
    return tensor

def tensor_to_string(tensor,pad_string ='_',sos=False, voc_list = ascii_letters + ' '):
    string = ""
    tensor_length = tensor.shape[0]

    for idx in range(0, tensor_length):
        max_pos = np.argmax(tensor[idx, ])
        if max_pos < len(voc_list):
            char = voc_list[max_pos]
        else:
            char = pad_string
        string = string + char
    if sos:
        return string[1:]
    else:
        return string
