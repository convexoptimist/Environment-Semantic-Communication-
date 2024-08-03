import os
import datetime
import sys
import shutil
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd
import ast
import types
import matplotlib.pyplot as plt
import torchvision.transforms as transf
import os
import datetime
import sys
import shutil
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optimizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np




def select_submatrix(tensor, row_range, col_range):
    """
    Returns a new tensor that contains only the elements of the input tensor that are within the specified row and
    column range. Elements outside the range are set to zero.
    """
    new_tensor = torch.zeros_like(tensor)  # create a new tensor of the same shape as the input tensor, initialized to zero
    new_tensor[row_range[0]:row_range[1], col_range[0]:col_range[1]] = tensor[row_range[0]:row_range[1], col_range[0]:col_range[1]]
    return new_tensor


class PositionDataset(Dataset):
    def __init__(self, csv_file, seq_len, transform = None):
        data = pd.read_csv(csv_file)
        data = data.reset_index(drop=True)
        self.input = data['unit3_rgb_seq'].values
        self.bbox1 = data['filtered_bbox_img1'].values
        self.bbox2 = data['filtered_bbox_img2'].values
        self.bbox3 = data['filtered_bbox_img3'].values
        self.bbox4 = data['filtered_bbox_img4'].values
        self.bbox5 = data['filtered_bbox_img5'].values
        self.labels = data['beam_index_5'].values
        self.n = seq_len
        self.transform =transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_tensor = torch.zeros((self.n,28,28))
        bbox_lst= [self.bbox1[index], self.bbox2[index], self.bbox3[index], self.bbox4[index], self.bbox5[index]]
        for _,inx in enumerate(ast.literal_eval(self.input[index])):
          img = Image.open('../../LeNet_on_Masks/unit3_masks/masks/' + inx[11:])
          img = np.array(img, dtype=np.uint8)
          bbox = bbox_lst[_]
          bbox= ast.literal_eval(bbox)
          col1 = bbox[0]
          row1 = bbox[1]
          col2 = bbox[2]
          row2 = bbox[3]
          img = select_submatrix(torch.tensor(img), (row1, row2), (col1, col2))
          if self.transform:
              img = self.transform(img)
          unique_elements = torch.unique(img)
          img = torch.where(img>0.00, 1.000, 0.000)
          input_tensor[_]=img
        label_tensor = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return input_tensor, label_tensor


##############################################################################################


# Define the LSTM-based model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 64)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        input_size = x.shape[2:]

        td_concat_size = (batch_size*time_steps,1,)+ input_size


        x = x.view(td_concat_size)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(batch_size, time_steps, -1)

        lstm_out, _ = self.lstm(x.permute(1, 0, 2))
        last_out = lstm_out[-1]
        output = self.fc(last_out)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)


img_resize = transf.Resize((28, 28))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
    img_resize,
      transf.ToTensor()]
)



test_file_path = 'test_unit3.csv'
test_dataset = PositionDataset(test_file_path, transform = proc_pipe,  seq_len=5)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model, loss function, and optimizer
model = LSTMClassifier(input_size=64, hidden_size=64, output_size=192).to(device)
loss_fn = nn.CrossEntropyLoss()
# model = model.cuda()

# Train the model

for epoch in range(1):
    # Validate the model
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ave_top5_acc = 0
    ind_ten = torch.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    top5_pred_out = []
    gt_beam = []
    model.load_state_dict(torch.load('saved_folder/LSTM_curr_beam_pred'))
    model=model.cuda()
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            num_correct += torch.sum(predictions == labels).item()
            num_total += len(labels)

            gt_beam.append(labels.detach().cpu().numpy()[0].tolist())

            top1_pred_out.append(predictions.detach().cpu().numpy()[0].tolist())
            sorted_out = torch.argsort(outputs, dim=1, descending=True)

            top_2_pred = torch.index_select(sorted_out, dim=1, index=ind_ten[0:2])
            top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0].tolist())

            top_3_pred = torch.index_select(sorted_out, dim=1, index=ind_ten[0:3])
            top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist() )

            top_5_pred = torch.index_select(sorted_out, dim=1, index=ind_ten[0:5])
            top5_pred_out.append(top_5_pred.detach().cpu().numpy()[0].tolist() )

            reshaped_labels = labels.reshape((labels.shape[0], 1))
            tiled_2_labels = reshaped_labels.repeat(1, 2)
            tiled_3_labels = reshaped_labels.repeat(1, 3)
            tiled_5_labels = reshaped_labels.repeat(1, 5)

            batch_top1_acc = torch.sum(predictions == labels, dtype=torch.float32)
            batch_top2_acc = torch.sum(top_2_pred == tiled_2_labels, dtype=torch.float32)
            batch_top3_acc = torch.sum(top_3_pred == tiled_3_labels, dtype=torch.float32)
            batch_top5_acc = torch.sum(top_5_pred == tiled_5_labels, dtype=torch.float32)

            ave_top1_acc += batch_top1_acc.item()
            ave_top2_acc += batch_top2_acc.item()
            ave_top3_acc += batch_top3_acc.item()


        print('Average Top-1 accuracy {}'.format( ave_top1_acc / num_total))
        print('Average Top-2 accuracy {}'.format( ave_top2_acc / num_total))
        print('Average Top-3 accuracy {}'.format( ave_top3_acc / num_total))

        accuracy = num_correct / num_total
        print(f"Test accuracy: {accuracy:.4f}")
        cur_accuracy  = accuracy


        file_to_save = 'LSTM_pred_beam_at_testing.csv'
        indx = np.arange(1, len(top1_pred_out)+1, 1)
        df1 = pd.DataFrame()
        df1['index'] = indx
        df1['link_status'] = gt_beam
        df1['top1_pred'] = top1_pred_out
        df1['top2_pred'] = top2_pred_out
        df1['top3_pred'] = top3_pred_out
        # df1.to_csv(file_to_save, index=False)