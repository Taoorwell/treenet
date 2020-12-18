from func import *
import os
from unet import *
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

tif_file = [r'../tiles/tile_WV3_Pansharpen_11_2016_{}.tif'.format(n+1) for n in range(25)]
mask_file = [r'../masks_single_trees/mask_single_{}.tif'.format(n+1) for n in range(25)]

#############
crowndataset = CrownDataset(tif_file=tif_file, mask_file=mask_file, m=200, n_random=250)
# for i in tqdm(range(len(crowndataset))):
#     sample = crowndataset[i]
#     print(sample['patch'].size(), sample['mask'].size(),
#           sample['location'])
#     if i == 10:
#         break

##############

# for i in tqdm(range(len(crowndataset))):
#     sample = crowndataset[i]
#     location = sample['location']
#     plt.scatter(x=location[0], y=location[1])
#     square = plt.Rectangle((location[0]-99, location[1]-99), 200, 200, ec='cyan', fc='none')
#     plt.gca().add_patch(square)
#     if i == 250:
#         break
# square = plt.Rectangle((0, 0), 1000, 1000, ec='red', fc='none')
# plt.gca().add_patch(square)
# plt.show()
##############################

unet = Unet(7, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    unet = nn.DataParallel(unet)
unet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(unet.parameters(), lr=0.001)
dataload = DataLoader(dataset=crowndataset, batch_size=10)
for b, sample_b in enumerate(dataload):
    patch = sample_b['patch'].to(device=device, dtype=torch.float32)
    mask = sample_b['mask'].to(device=device, dtype=torch.long)
    optimizer.zero_grad()
    out = unet(patch)
    loss = criterion(out, mask)
    loss.backward()
    optimizer.step()
    print(b, loss)

#######################
# torch.save(unet.state_dict(), '.checkpoints/unet.pth')
# unet.load_state_dict(torch.load('.checkpoints/unet-2.pth', map_location=device))
# sample = crowndataset[0]
# patch = sample['patch'].to(device=device, dtype=torch.float32).reshape(1, 7, 200, 200)
# out = unet(patch)
# out = torch.argmax(out, dim=1)
# out = out.detach().numpy()
# print(out, out.shape)
# ax = plt.subplot2grid((1, 3), (0, 0))
# ax.imshow(patch.detach().numpy()[0].transpose((1, 2, 0))[:, :, 1])
# ax1 = plt.subplot2grid((1, 3), (0, 2))
# ax1.imshow(out.reshape(200, 200))
# ax2 = plt.subplot2grid((1, 3), (0, 1))
# ax2.imshow(sample['mask'].detach().numpy())
# plt.show()


