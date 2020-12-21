from func import *
import os
from unet import *
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

tif_file = [r'../tiles/tile_WV3_Pansharpen_11_2016_{}.tif'.format(n+1) for n in range(25)]
mask_file = [r'../masks_single_trees/mask_single_{}.tif'.format(n+1) for n in range(25)]

tif_file_eval = [r'../tiles/tile_WV3_Pansharpen_11_2016_{}.tif'.format(n+1) for n in range(25, 30)]
mask_file_eval = [r'../masks_single_trees/mask_single_{}.tif'.format(n+1) for n in range(25, 30)]

tif_files = [get_image_data(file) for file in tif_file]
mask_files = [get_raster_info(file) for file in mask_file]
tif_file_evals = [get_image_data(file) for file in tif_file_eval]
mask_file_evals = [get_raster_info(file) for file in mask_file_eval]


#############
crowndataset = CrownDataset(tif_file=tif_files, mask_file=mask_files, m=200, n_random=300)

crowndataset_eval = CrownDataset(tif_file=tif_file_evals, mask_file=mask_file_evals, m=200, n_random=250)


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
writer = SummaryWriter(comment='lr_{}_batch_{}_sample_{}'.format(0.8, 30, 250))
unet = Unet(7, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    unet = nn.DataParallel(unet)
unet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.0001, momentum=0.8)
dataload = DataLoader(dataset=crowndataset, batch_size=30)
dataload_eval = DataLoader(dataset=crowndataset_eval, batch_size=25)

# writer.add_graph(model=unet)
for epoch in range(50):
    to_loss = 0
    for sample_b in tqdm(dataload):
        patch = sample_b['patch'].to(device=device, dtype=torch.float32)
        mask = sample_b['mask'].to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        out = unet(patch)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()
        to_loss += loss
    epoch_loss = to_loss / len(dataload)
    writer.add_scalar('train_loss', epoch_loss)
    tot = evaluation(unet, dataload_eval, device)
    writer.add_scalar('eval_loss', tot)
    for tag, value in unet.named_parameters():
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())
    print('Epoch:{} Train Finish'.format(epoch+1), 'Average Train Loss:{}'.format(epoch_loss),
          'Evaluation Loss:{}'.format(tot))
    torch.save(unet.module.state_dict(), '.checkpoints/unet-{}'.format(epoch+1))
writer.close()


#######################
# torch.save(unet.state_dict(), '.checkpoints/unet.pth')
# unet = Unet(7, 2)
# unet.load_state_dict(torch.load('.checkpoints/unet-10.pth', map_location=torch.device('cpu')))
# # tot = evaluation(unet, dataload_eval, device)
# # print(tot)
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


