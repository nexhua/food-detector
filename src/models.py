from model import Model
from model_definitions import get_generators, get_model

simple_v1 = Model(name='simple_v1',
                  validation_split=0.2, 
                  batch_size=32, 
                  optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'], 
                  epochs=5)

simple_v1.model = get_model(simple_v1)
data_generator, test_generator = get_generators(simple_v1)
simple_v1.data_generator = data_generator
simple_v1.test_generator = test_generator
