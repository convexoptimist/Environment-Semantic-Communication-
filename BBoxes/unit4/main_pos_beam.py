

def main():

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



    ########################################################################
    ######################### Hyperparameters ##############################
    ########################################################################
    
    # Hyperparameters for our network
    input_size = 2
    node = 512
    output_size = 192
    
    
    # Training Hyper-parameters
    val_batch_size = 1
    train_size = [1]
    
    ########################################################################    
    ########################################################################
    

                           
    ############################################    
    # Define a custom dataset class that reads from a CSV file          
    class CustomDataset(Dataset):
        def __init__(self, csv_file, transform=None):
            self.data = pd.read_csv(csv_file)
            self.transform = transform
    
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):            
            center_cord = ast.literal_eval(self.data.loc[idx, 'bbox'])
            label = self.data.loc[idx, 'beam_index']
    
            center_cord= torch.tensor(center_cord)
            label = torch.tensor(label, dtype=torch.long)
            return center_cord, label

    ########################################################################
    
    ########################################################################
    ########################### Data pre-processing ########################
    ########################################################################
    proc_pipe = transf.Compose(
        [

         transf.ToTensor()
        ]
    )
    train_dir = 'bbox_unit4_beam_pred_train.csv'
    val_dir = 'bbox_unit4_beam_pred_val.csv'
    test_dir = 'bbox_unit4_beam_pred_test.csv'



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

    ########################################################################
    ########################################################################
    
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = NN_beam_pred(input_size, output_size)    


    ########################################################################
    ########################################################################
    ################### Load the model checkpoint ##########################    


    net_name = "saved_folder/2-layer_nn_beam_pred"
    running_top1_acc = []  
    running_top2_acc = []
    running_top3_acc = []

    model.load_state_dict(torch.load(net_name))
    model.eval() 
    net = model.cuda()   
    print()
    print()
    print('Loading model and testing')
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    total_count = 0

    gt_beam = []
    for val_count, (pos_data, beam_val) in enumerate(test_loader):
        net.eval()
        data = pos_data.type(torch.Tensor)  
        x = data.cuda()                    
        labels = beam_val.type(torch.LongTensor)   
        # opt.zero_grad()
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
        top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist()  )
            
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
   
    # print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
    print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
    print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
    print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))

    print("Saving the predicted value in a csv file")
    file_to_save = 'best_epoch_eval.csv'
    indx = np.arange(1, len(top1_pred_out)+1, 1)
    df2 = pd.DataFrame()
    df2['index'] = indx                
    df2['link_status'] = gt_beam
    df2['top1_pred'] = top1_pred_out
    df2['top2_pred'] = top2_pred_out
    df2['top3_pred'] = top3_pred_out
    # df2.to_csv(file_to_save, index=False)   


    
if __name__ == "__main__":
    #run()
    main()