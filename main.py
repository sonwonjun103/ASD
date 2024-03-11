import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from torch.utils.data import DataLoader

from data.dataset import CustomDataset
from data.load_path import index_label

from models.Model import Model 
from train.trainer import train_loop

from utils.seed import seed_everything
from utils.parser import set_parser
from utils.transform import get_aug

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print(f"Device : {device}")
    
    seed_everything(args.seed)
    
    axial_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Axial\\T1_axial_train.xlsx")
    sagittal_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Sagittal\\T1_sagittal_train.xlsx")
    coronal_train = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Coronal\\T1_coronal_train.xlsx")

    axial_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Axial\\T1_axial_test.xlsx")
    sagittal_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Sagittal\\T1_sagittal_test.xlsx")
    coronal_test = pd.read_excel(f"D:\\새 폴더\\Entropy\\T1\\Coronal\\T1_coronal_test.xlsx")
    
    train_index, train_label, train_asd, train_tc = index_label(axial_train)
    test_index, test_label, test_asd, test_tc = index_label(axial_test)

    print(f"Train ASD patient : {train_asd}")
    print(f"Train TC patient : {train_tc}")

    print(f"Test ASD patient : {test_asd}")
    print(f"Test TC patient : {test_tc}")
    
    train_transform, test_transform = get_aug(args)
    
    # Make Dataset & DataLoader
    train_dataset = CustomDataset(train_index, axial_train, sagittal_train, coronal_train, train_label, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

    test_dataset = CustomDataset(test_index, axial_test, sagittal_test, coronal_test, test_label, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)
    
    # define model
    model = Model(2048).to(device)
    model = nn.DataParallel(model).to(device)
    
    # set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    weight = torch.Tensor([train_tc, train_asd]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(weight).to(device)
    
    #Train
    train_loss, train_acc = train_loop(train_dataloader, test_dataloader, model, optimizer, loss_fn, test_label, device)    

if __name__=='__main__':
    args = set_parser()
    main(args)