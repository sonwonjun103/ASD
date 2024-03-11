import numpy as np

def index_label(excel):
    label=[]
    index=[]

    asd=0
    tc=0

    for i in range(len(excel)):
        path = excel.iloc[i].values[-1]
        if 'ASD' in path:
            label.append(np.array([0,1]))
            asd+=1
        else:
            label.append(np.array([1,0]))
            tc+=1

        index.append(i)

    return index, label, asd, tc