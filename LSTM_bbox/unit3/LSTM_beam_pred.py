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




# Define a custom dataset to read in the data from CSV files
class PositionDataset(Dataset):
    def __init__(self, csv_file, seq_len):
        data = pd.read_csv(csv_file, header=None)
        self.inputs = data.iloc[:, 3:8].values
        self.labels = data.iloc[:, 2].values
        self.n = seq_len
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_tensor = torch.zeros((self.n,2))
        for i,s in enumerate(self.inputs[index]):
            data = s 
            # print('data',data)
            bbox_data = ast.literal_eval(data)            
            input_tensor[i] = torch.tensor(bbox_data, requires_grad=False)
        #print(	input_tensor)
        label_tensor = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return input_tensor, label_tensor
        


# Define the LSTM-based model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        #lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        lstm_out, _ = self.lstm(input_seq.permute(1, 0, 2))
        last_out = lstm_out[-1]
        output = self.fc(last_out)
        return output
        

# Create datasets and data loaders
test_dataset = PositionDataset('bbox_test_unit3_shuffled2_v2_new.csv',  seq_len=5)
test_loader = DataLoader(test_dataset, batch_size = 1)



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model, loss function, and optimizer
model = LSTMClassifier(input_size=2, hidden_size=64, output_size=192).to(device)
loss_fn = nn.CrossEntropyLoss()





net_name = "saved_folder/LSTM_curr_beam_pred"

print()
print()
print()
print('Loading the model and validating')


for epoch in range(1):
    # Test the model
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

    model.load_state_dict(torch.load(net_name)) 
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
            ave_top5_acc += batch_top5_acc.item()
           

     
        print('Average Top-1 accuracy {}'.format( ave_top1_acc / num_total))
        print('Average Top-2 accuracy {}'.format( ave_top2_acc / num_total))
        print('Average Top-3 accuracy {}'.format( ave_top3_acc / num_total))      
             
        accuracy = num_correct / num_total
        print(f"Test accuracy: {accuracy:.4f}")
        cur_accuracy  = accuracy


        
        file_to_save = 'LSTM_pred_beam_after_{epoch+1}th_epoch.csv'
        indx = np.arange(1, len(top1_pred_out)+1, 1)
        df1 = pd.DataFrame()
        df1['index'] = indx                
        df1['link_status'] = gt_beam
        df1['top1_pred'] = top1_pred_out
        df1['top2_pred'] = top2_pred_out
        df1['top3_pred'] = top3_pred_out
        # df1.to_csv(file_to_save, index=False)  
        

