from network import initialize_network, train_network
from perprocessing import preprocessing_raw_data

if __name__ == '__main__':
    dataset = preprocessing_raw_data()
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network=network, train=dataset, l_rate=0.005, n_epoch=3000, n_outputs=n_outputs)
    for layer in network:
        print(layer)
