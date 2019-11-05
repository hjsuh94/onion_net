from onion_models import OnionEncoder, OnionDecoder, OnionNet
from onion_dataset import OnionDataset
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

onion_encoder = OnionEncoder()
onion_encoder.load_state_dict(torch.load("onionencoder.pth"))
onion_encoder.eval()
onion_decoder = OnionDecoder()
onion_decoder.load_state_dict(torch.load("oniondecoder.pth"))
onion_decoder.eval()
onion_net = OnionNet()
onion_net.load_state_dict((torch.load("onion_net.pth")))
onion_net.eval()

def evaluate_model(image, ui, encoder, decoder, net):
    image_i = Variable(image).cuda()
    u = Variable(ui).cuda().float()

    z_i = encoder(image_i)
    z_i = z_i.view(batch_size, 221).float()
    z_iu = torch.cat((z_i, u), 1)

    z_f = onion_net(z_iu)
    z_f = z_f.view(batch_size, 1, 13, 17)
    image_fh = oniondecoder(z_f)

    return image_fh

dataset = OnionDataset("data.csv", "proc", os.getcwd())

index = 3

image_i = dataset[index]['image_i']
image_f = dataset[index]['image_f']
u = dataset[index]['u']

image_i_disp = torchvision.transforms.functional.to_pil_image(image_i)
image_f_disp = torchvision.transforms.functional.to_pil_image(image_f)
image_fhat = torchvision.transforms.functional.to_pil_image(\
                 evaluate_model(image_i, u, onion_encoder, onion_decoder, onion_net))

plt.figure()
plt.imshow(image_i_disp)
plt.show()

plt.figure()
plt.imshow(image_fhat)
plt.show()

plt.figure()
plt.imshow(image_f_disp)
plt.show()



