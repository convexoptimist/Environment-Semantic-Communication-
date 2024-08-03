



import os
import datetime
import sys
import shutil
import torch as t
import torch
import torch.cuda as cuda
import torch.optim as optimizer
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transf
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image




########################################################################
######################### Hyperparameters ##############################
########################################################################

# Hyperparameters for our network


#Hyper-parameters
val_batch_size = 1
train_size = [1]

########################################################################
########################################################################


def select_submatrix(tensor, row_range, col_range):
    """
    Returns a new tensor that contains only the elements of the input tensor that are within the specified row and
    column range. Elements outside the range are set to zero.
    """
    new_tensor = torch.zeros_like(tensor)  # create a new tensor of the same shape as the input tensor, initialized to zero
    new_tensor[row_range[0]:row_range[1], col_range[0]:col_range[1]] = tensor[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    return new_tensor






def create_samples(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    for idx, row in f.iterrows():
        img_paths = row.values
        data_samples.append(img_paths)

    # print(data_samples)
    return data_samples


############################################
# Define a custom dataset class that reads from a CSV file
class CustomDataset(Dataset):
    def __init__(self,root_dir, nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_samples(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_name = sample[1]
        beam_index = sample[-1]
        bbox = sample[2]
        img = Image.open('../unit3_masks/masks/' + file_name[11:])
        img = np.array(img, dtype=np.uint8)
        bbox= ast.literal_eval(bbox)
        col1 = bbox[0]
        row1 = bbox[1]
        col2 = bbox[2]
        row2 = bbox[3]
        img = select_submatrix(torch.tensor(img), (row1, row2), (col1, col2))
        if self.transform:
            img = self.transform(img)
        unique_tensor = np.unique(img)
        img = torch.where(img>0.00, 1.000, 0.000)
        return (img,beam_index)



########################################################################
########################################################################
########################### Data pre-processing ########################
########################################################################
img_resize = transf.Resize((28, 28))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
    img_resize,
      transf.ToTensor()]
)

test_dir = 'unit3_beam_pred_test.csv'


test_loader = DataLoader(CustomDataset(test_dir, transform=proc_pipe),
                        batch_size=val_batch_size,
                        shuffle=False)

########################################################################
########################################################################
##################### Model Definition #################################
########################################################################

class NN_beam_pred(nn.Module):
    def __init__(self, num_features, num_output):
        super(NN_beam_pred, self).__init__()

        self.layer_1 = nn.Linear(num_features, node)
        self.layer_2 = nn.Linear(node, node)
        self.layer_3 = nn.Linear(node, node)
        self.layer_out = nn.Linear(node, num_output)

        self.relu = nn.ReLU()



    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)


class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 256)
        self.fc4   = nn.Linear(256, 256)
        self.fc5   = nn.Linear(256, 192)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)


########################################################################
########################################################################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = NN_beam_pred(input_size, output_size)
model = LeNet()

########################################################################
#################### Model Training ####################################
########################################################################
val_acc = []
with cuda.device(0):
    top_1 = np.zeros( (1,len(train_size)) )
    top_2 = np.zeros( (1,len(train_size)) )
    top_3 = np.zeros( (1,len(train_size)) )
    acc_loss = 0
    itr = []
    for idx, n in enumerate(train_size):
        print('```````````````````````````````````````````````````````')
        print('Training size is {}'.format(n))
        # Build the network:
        net = model.cuda()
        # layers = list(net.children())

        #  Optimization parameters:
        criterion = nn.CrossEntropyLoss()
        count = 0
        net_name = "saved_folder/2-layer_nn_beam_pred"
        running_loss = []
        running_top1_acc = []
        running_top2_acc = []
        running_top3_acc = []
        best_accuracy = 0

        for epoch in range(1):
            print('Epoch No. ' + str(epoch + 1))
            skipped_batches = 0

            print('Start validation')
            ave_top1_acc = 0
            ave_top2_acc = 0
            ave_top3_acc = 0
            ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
            top1_pred_out = []
            top2_pred_out = []
            top3_pred_out = []
            total_count = 0

            gt_beam = []
            net.load_state_dict(torch.load(net_name))
            net=net.cuda()
            net.eval()
            for val_count, (pos_data, beam_val) in enumerate(test_loader):
                net.eval()
                data = pos_data.type(torch.Tensor)
                x = data.cuda()
                labels = beam_val.type(torch.LongTensor)
                labels = labels.cuda()
                gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
                total_count += labels.size(0)
                out = net.forward(x)
                _, top_1_pred = t.max(out, dim=1)
                top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0].tolist())
                sorted_out = t.argsort(out, dim=1, descending=True)

                top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
                top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0].tolist())

                top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:3])
                top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist() )

                reshaped_labels = labels.reshape((labels.shape[0], 1))
                tiled_2_labels = reshaped_labels.repeat(1, 2)
                tiled_3_labels = reshaped_labels.repeat(1, 3)

                batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
                batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
                batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)

                ave_top1_acc += batch_top1_acc.item()
                ave_top2_acc += batch_top2_acc.item()
                ave_top3_acc += batch_top3_acc.item()

            print("total test examples are", total_count)
            running_top1_acc.append(ave_top1_acc / total_count)  # (batch_size * (count_2 + 1)) )
            running_top2_acc.append(ave_top2_acc / total_count)
            running_top3_acc.append(ave_top3_acc / total_count)  # (batch_size * (count_2 + 1)))

            print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
            print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
            print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
            print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))



            cur_accuracy  = running_top1_acc[-1]


            print("Saving the predicted value in a csv file")
            file_to_save = 'top1_pred_beam_val_after_{epoch+1}th_epoch.csv'
            indx = np.arange(1, len(top1_pred_out)+1, 1)
            df1 = pd.DataFrame()
            df1['index'] = indx
            df1['link_status'] = gt_beam
            df1['top1_pred'] = top1_pred_out
            df1['top2_pred'] = top2_pred_out
            df1['top3_pred'] = top3_pred_out
            # df1.to_csv(file_to_save, index=False)

            # LR_sch.step()
        top_1[0,idx] = running_top1_acc[-1]
        top_2[0,idx] = running_top2_acc[-1]
        top_3[0,idx] = running_top3_acc[-1]




