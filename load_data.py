def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle("cifar-10-batches-py\data_batch_1")
dict = unpickle("cifar-10-batches-py\data_batch_2")
dict = unpickle("cifar-10-batches-py\data_batch_3")
dict = unpickle("cifar-10-batches-py\data_batch_4")
dict = unpickle("cifar-10-batches-py\data_batch_5")
dict = unpickle("cifar-10-batches-py\test_batch")