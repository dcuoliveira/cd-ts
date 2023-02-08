import pickle

def save_pkl(data,
             path):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def load_pkl(path):

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
    return data