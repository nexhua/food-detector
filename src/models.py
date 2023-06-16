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


cnn_v1 = Model(name='cnn_v1', validation_split=0.2, batch_size=32, optimizer='adam',
               loss='categorical_crossentropy', metrics=['accuracy'], epochs=8)
cnn_v1.model = get_model(cnn_v1)
data_generator, test_generator = get_generators(cnn_v1)
cnn_v1.data_generator = data_generator
cnn_v1.test_generator = test_generator

cnn_v2 = Model(name='cnn_v2', validation_split=0.2, batch_size=32, optimizer='adam',
               loss='categorical_crossentropy', metrics=['accuracy'], epochs=8)
cnn_v2.model = get_model(cnn_v2)
data_generator, test_generator = get_generators(cnn_v2)
cnn_v2.data_generator = data_generator
cnn_v2.test_generator = test_generator


cnn_v3 = Model(name='cnn_v3', validation_split=0.2, batch_size=32, optimizer='adam',
               loss='categorical_crossentropy', metrics=['accuracy'], epochs=10)
cnn_v3.model = get_model(cnn_v3)
data_generator, test_generator = get_generators(cnn_v3)
cnn_v3.data_generator = data_generator
cnn_v3.test_generator = test_generator

fine_tuned_v1 = Model(name='fine_tuned_v1', validation_split=0.2, batch_size=32, optimizer='adam',
               loss='categorical_crossentropy', metrics=['accuracy'], epochs=5)
fine_tuned_v1.model = get_model(fine_tuned_v1)
data_generator, test_generator = get_generators(fine_tuned_v1)
fine_tuned_v1.data_generator = data_generator
fine_tuned_v1.test_generator = test_generator

fine_tuned_v2 = Model(name='fine_tuned_v2', validation_split=0.2, batch_size=32, optimizer='adam',
               loss='categorical_crossentropy', metrics=['accuracy'], epochs=5)
fine_tuned_v2.model = get_model(fine_tuned_v2)
data_generator, test_generator = get_generators(fine_tuned_v2)
fine_tuned_v2.data_generator = data_generator
fine_tuned_v2.test_generator = test_generator