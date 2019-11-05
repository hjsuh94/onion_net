import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from onion_dataset import OnionDataset
from onion_models import OnionEncoder, OnionDecoder, OnionNet
from torchsummary import summary
from itertools import chain
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/onion_net_11')

import warnings
warnings.filterwarnings("ignore")

onion_dataset = OnionDataset("data.csv", "proc", os.getcwd())
dataloader = DataLoader(onion_dataset, batch_size=10, \
                        shuffle=True, num_workers=16)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 100, 124)
    return x 

num_epochs = 1000
batch_size = 512
learning_rate = 1e-3

img_trasnform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = OnionDataset("data.csv", "proc", os.getcwd())
dataloader = DataLoader(dataset, batch_size=batch_size,\
                        shuffle=True, num_workers=16, drop_last = True)

testbatch = [0, 1, 2, 20, 25]
testimage_i = []
testimage_f = []
testu = []

for i in testbatch:
    testimage_i.append(onion_dataset[i]['image_i'])
    testimage_f.append(onion_dataset[i]['image_f'])
    testu.append(onion_dataset[i]['u'])

onionencoder = OnionEncoder().cuda()
oniondecoder = OnionDecoder().cuda()
onion_net = OnionNet().cuda()

summary(onionencoder, (1, 100, 124))
summary(oniondecoder, (1, 13, 17))
summary(onion_net, (1, 225))

params = chain(onionencoder.parameters(), onion_net.parameters(), oniondecoder.parameters())
params = chain(onionencoder.parameters(), onion_net.parameters(), oniondecoder.parameters())


optimizer = torch.optim.Adam(params, lr=learning_rate,\
                             weight_decay = 1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    onionencoder.train()
    oniondecoder.train()
    onion_net.train()

    running_loss = 0.0
    
    for data in dataloader:
        

        image_i = Variable(data['image_i']).cuda()
        image_f = Variable(data['image_f']).cuda()

        u = Variable(data['u']).cuda().float()

        #print(image_i.shape)#
        z_i = onionencoder(image_i)
        #print(z_i.shape)
        z_i = z_i.view(batch_size, 221).float()
        z_iu = torch.cat((z_i, u), 1)

        z_f = onion_net(z_iu)
        z_f = z_f.view(batch_size, 1, 13, 17)

        image_fh = oniondecoder(z_f)
        #image_fh = (image_fh - torch.min(image_fh)) / (torch.max(image_fh) - torch.min(image_fh))

        loss = torch.norm(torch.abs(image_f - image_fh))

        running_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch[{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    scheduler.step(running_loss)

    writer.add_scalar('training_loss', loss.item(), epoch)

    onionencoder.eval()
    oniondecoder.eval()
    onion_net.eval()

    for i in range(len(testbatch)):
        image_i = testimage_i[i].cuda()
        image_f = testimage_f[i].cuda()
        u = torch.from_numpy(testu[i]).cuda().float()

        z_i = onionencoder(torch.unsqueeze(image_i, 0))
        z_i = z_i.view(1, 221).float()

        z_iu = torch.cat((z_i, torch.unsqueeze(u,0)), 1)

        z_f = onion_net(z_iu)
        z_f = z_f.view(1, 1, 13, 17)
        image_fh = oniondecoder(z_f)

        normalize = torchvision.transforms.Normalize(mean=[-0.25], std=[0.25])
        image_fh = normalize(image_fh[0])

        # Threshold transform? 
        image_fh = (image_fh - torch.min(image_fh)) / (torch.max(image_fh) - torch.min(image_fh))

        writer.add_image(str(epoch) + "_" + str(i) + "i", image_i, dataformats='CHW')
        writer.add_image(str(epoch) + "_" + str(i) + "f", image_f, dataformats='CHW')
        writer.add_image(str(epoch) + "_" + str(i) + "fh", image_fh, dataformats='CHW')
    

torch.save(onionencoder.state_dict(), 'onionencoder.pth')
torch.save(oniondecoder.state_dict(), 'oniondecoder.pth')
torch.save(onion_net.state_dict(), 'onion_net.pth')

        



