from python_gdal import *
import os
import matplotlib.pyplot as plt
from unet import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

file = r'D:/Repos/temp'
tif_file = [r'/tiles/tile_WV3_Pansharpen_11_2016_{}.tif'.format(n+1) for n in range(25)]
mask_file = [r'/masks_single_trees/mask_single_{}.tif'.format(n+1) for n in range(25)]


def get_image_data(file):
    bands_data = get_raster_info(file)
    image_data = norma_data(bands_data, norma_methods='min-max')
    return image_data


# image_data = get_raster_info(file+tif_file[0])
# print(image_data.shape)


class CrownDataset(Dataset):
    def __init__(self, tif_file, mask_file, n_random):
        self.tif_file = tif_file
        self.mask_file = mask_file
        self.n_random = n_random

    def __len__(self):
        return len(self.tif_file) * self.n_random

    def __getitem__(self, item):
        i = item // self.n_random
        image_data = get_image_data(os.path.join(file + self.tif_file[i]))
        mask_data = get_raster_info(os.path.join(file + self.mask_file[i]))
        location = random_sample(self.n_random)
        new_item = item % self.n_random
        (h, w) = location[new_item]
        patch = torch.from_numpy(image_data[h-99: h+101, w-99: w+101].transpose([2, 0, 1]))
        mask = torch.from_numpy(mask_data[h-99: h+101, w-99: w+101][:, :, 0])
        sample = {'patch': patch, 'mask': mask, 'location': (h, w)}
        return sample


def random_sample(n):
    x = np.random.randint(99, 899, (n,))
    y = np.random.randint(99, 899, (n,))
    location = [(h, w) for h, w in zip(x, y)]
    return location


def show_sample(sample):
    patch = np.array(sample['patch']).transpose((1, 2, 0))[:, :, 3:6]
    mask = np.array(sample['mask'])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(patch)
    ax2.imshow(mask)
    plt.title('Location:{}'.format(sample['location']))
    plt.show()


#############
crowndataset = CrownDataset(tif_file=tif_file, mask_file=mask_file, n_random=250)
# for i in tqdm(range(len(crowndataset))):
#     sample = crowndataset[i]
#     print(sample['patch'].size(), sample['mask'].size(),
#           sample['location'])
#     if i == 10:
#         break

##############

#for i in tqdm(range(len(crowndataset))):
#    sample = crowndataset[i]
#    location = sample['location']
#    plt.scatter(x=location[0], y=location[1])
#    square = plt.Rectangle((location[0]-99, location[1]-99), 200, 200, ec='cyan',
#                           fc='none')
#    plt.gca().add_patch(square)
#    if i == 500:
#        break
# square = plt.Rectangle((0, 0), 1000, 1000, ec='red', fc='none')
# plt.gca().add_patch(square)
# plt.show()
##############################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = Unet(7, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)
dataload = DataLoader(dataset=crowndataset, batch_size=10)
for b, sample_b in enumerate(dataload):
    patch = sample_b['patch'].to(device=device, dtype=torch.float32)
    mask = sample_b['mask'].to(device=device, dtype=torch.long)
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


