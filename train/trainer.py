import torch
import time

from utils.normalize import *
from utils.seed import seed_everything

#from test.test import test_loop

def train_loop(args, train_dataloader, test_dataloader, model, optimizer, loss_fn, test_label, device):
    seed_everything(args.seed)

    iterations = len(train_dataloader)
    size = len(train_dataloader.dataset)

    train_loss=[]
    train_acc=[]

    model.train()

    for epoch in range(args.epochs):
        print(f"Start epoch : {epoch+1}")
        epoch_start = time.time()
        epoch_correct=0
        epoch_loss=0
        for batch, (X,y) in enumerate(train_dataloader):
            X=torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])).to(device).float()
            y=y.to(device).float()

            outputs = model(X)
            correct=0

            loss = loss_fn(outputs, y)
            correct+=(outputs.argmax(1)==y.argmax(1)).detach().cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            epoch_correct += correct
        
            if batch % 10 == 0:
                print(f"Loss : {loss:>.5f} Correct : [{epoch_correct}/{size}] Batch : [{batch}/{iterations}]")

        train_loss.append(epoch_loss)
        train_acc.append(epoch_correct)

        print(f"Train\n Loss : {epoch_loss/batch:>.5f} Correct : {(epoch_correct/size * 100):.2f}% [{epoch_correct}/{size}]")
        epoch_end = time.time()

        model_path = f"./model_parameters/model_focal_{epoch+1}.pt"
        torch.save(model.state_dict(), model_path)

        #sens, spec, auc, acc, best_thresh = test_loop(test_dataloader, model, model_path, test_label, device, epoch)

        #print(f"Test\n Sens : {sens:.>3f} Spec : {spec:.>3f} AUC : {auc:.>3f} ACC : {acc:.>3f} thresholds : {best_thresh:.>3f}")
        print(f"Epoch time : {(epoch_end - epoch_start) // 60}min {(epoch_end - epoch_start) % 60}sec")
        print(f"End epoch {epoch+1}\n")

    return train_loss, train_acc