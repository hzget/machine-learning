import network

network.train_model(layers=[784, 30, 10], epochs=30, mini_batch_size=10, eta=3.0)
