import torch
import matplotlib.pyplot as plt
from sklearn.metrics import *

from utils.seed import *
from utils.normalize import *

def test_loop(args, dataloader, model, model_path, label, device, epoch):
    seed_everything(args.seed)

    model.eval()
    model.load_state_dict(torch.load(model_path))

    target=[] 

    pred=[]
    correct=0
    with torch.no_grad():
        for X,y in dataloader:
            X = torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])).to(device).float()
            y = y.to(device).float()

            for t in y:
                target.append(t.detach().cpu().tolist()[1])

            outputs = model(X)
            
            for o in outputs:
                pred.append(o.detach().cpu().tolist()[1])
            print(outputs)
            correct+=(outputs.argmax(1)==y.argmax(1)).detach().cpu().sum()
    
    fpr, tpr, thresholds = roc_curve(target, pred)

    J=tpr-fpr
    idx = np.argmax(J)

    best_thresh = thresholds[idx]

    sens, spec = tpr[idx], 1-fpr[idx]

    asd, tc = 20,20
    acc = (sens*asd +spec * tc) / (asd+tc)
    auc = roc_auc_score(target, pred)

    plt.title("ROC CURVE")
    plt.plot([0,1], [0,1], linestyle='--', markersize=0.01, color='black')
    plt.plot(fpr, tpr, marker='.', color='black', markersize=0.05)
    plt.scatter(fpr[idx], tpr[idx], marker='+', s=100, color='r',
                label = 'Best threshold = %.3f, \nSensitivity : %.3f (%d / %d), \nSpecificity = %.3f (%d / %d), \nAUC = %.3f , \nACC = %.3f (%d / %d)' % (best_thresh, sens, (sens*asd), asd, spec, (spec*tc), tc, auc, acc, (sens*asd+spec*tc), 40))
    plt.legend()
    plt.savefig(f"./roc_curve/roc_curve_focal_{epoch+1}.png")

    plt.clf()

    return sens, spec, auc, acc, best_thresh
