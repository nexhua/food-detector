class Model:
    def __init__(self, name, validation_split, batch_size, optimizer, loss, metrics, epochs):
        self.name = name
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.model = None
        self.data_generator = None
        self.test_generator = None
