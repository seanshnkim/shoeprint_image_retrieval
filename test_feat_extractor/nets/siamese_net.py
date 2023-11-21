import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_network, end_layer=7, embdim=500):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = embedding_network
        
        if end_layer == 7:
            self.fc1 = nn.Sequential(
                nn.Linear(1024*7*7, embdim),
                nn.ReLU(inplace=True),

                nn.Linear(embdim, embdim),
                nn.ReLU(inplace=True),

                nn.Linear(embdim, 5))
            
        elif end_layer == 5:
            self.fc1 = nn.Sequential(
                nn.Linear(256*25*25, embdim),
                nn.ReLU(inplace=True),

                nn.Linear(embdim, embdim),
                nn.ReLU(inplace=True),

                nn.Linear(embdim, 5))

    def forward_once(self, x):
        output = self.embedding_network(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, *inputs):
        input1, input2, input3 = inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        
        return output1, output2, output3