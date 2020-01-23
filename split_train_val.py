
import numpy as np
import glob


def get_files_names(data_path,name_file):
    train_val_path = str(data_path / 'train_val{}'/ 'images').format(name_file)+ "/*.npy"
    fpath_list = sorted(glob.glob(train_val_path))

    
    dataset_size = len(fpath_list)
    indices = list(range(dataset_size))

    split = int(np.floor(0.2 * dataset_size))


    if 1 :
        #np.random.seed(1337)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    print(len(train_indices), len(valid_indices))
    #suffle the indices of 0-915 and then train_indices 0-80% and val_ind 80-100%
    
    
    train_file_names=[]
    val_file_names=[]
    
    for i in train_indices:
        train_file_names.append(fpath_list[i])
      
    for i in valid_indices:
        val_file_names.append(fpath_list[i])
        
    #print((train_file_names),(val_file_names)) 
        
    return train_file_names, val_file_names


if __name__ == '__main__':
    ids = get_split(data_path,name_file)