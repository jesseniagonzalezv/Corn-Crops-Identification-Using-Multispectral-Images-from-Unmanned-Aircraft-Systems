from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np


def get_split_out(data_path,name_file, fold, num_splits=5):
    #data_path=Path(path)

    train_path = data_path / name_file / 'images'

    train_file_names = np.array(sorted(list(train_path.glob('*npy'))))

    kf = KFold(n_splits=num_splits, random_state=2019,shuffle=True)
    


    ids = list(kf.split(train_file_names))

    train_ids, val_ids = ids[fold]

    if fold == -1:
        return train_file_names, train_file_names
    else:
        return train_file_names[train_ids], train_file_names[val_ids]


#if __name__ == '__main__':
#    train_file_names,val_file_names = get_split('data_HR','data',1)


def percent_split(train_val_100percent, percent = 1): 
    
    fpath_list = train_val_100percent


    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))
    percent = int(np.floor(percent * dataset_size))
    if 1 :
        np.random.seed(2019)
        np.random.shuffle(indices)        

    extra_indices, train_indices_split = indices[percent:], indices[:percent]
    print(dataset_size,len(train_indices_split), len(extra_indices))
    
   
    return train_val_100percent[extra_indices],train_val_100percent[train_indices_split] #
    
    
def get_split_in(train_file_names, fold, num_splits=5):
 
    kf = KFold(n_splits=num_splits, random_state=2019,shuffle=True)
    #kf = KFold(n_splits=num_splits, random_state=20018)


    ids = list(kf.split(train_file_names))

    train_ids, val_ids = ids[fold]

    if fold == -1:
        return train_file_names, train_file_names
    else:
        return train_file_names[train_ids], train_file_names[val_ids]
    
#if __name__ == '__main__':
#    train_file_names_2,val_file_names_2 = get_split_in(train_file_names,0)