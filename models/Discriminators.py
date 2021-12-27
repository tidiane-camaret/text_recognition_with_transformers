import torch.nn as nn
import torch

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
