# Japanese-handwriting-recognizer

A Transfer Learning based approach to recognizing the handwritten Hiragana characters of the Kusushiji-MNIST(KMNIST) dataset.Kuzushiji-MNIST is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format. Since MNIST restricts us to 10 classes, one character is chosen to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.

Both the train and test datasets are passed through ResNet50 pretrained on the ImageNet dataset, with the final dense layer removed. These bottleneck values are stored in resnet_features_train.npz and resnet_features_test.npz respectively. 

The sequential model with a GlobalAveragePooling2D,Dropout and Dense layer is then trained with the bottleneck train values.Finally, it is evaluated on the bottleneck test values.

The model with the lowest validation loss is stored by Checkpointer in model.best.hdf5. 

Due to computational limitations, the model is trained on 2000 images and tested on 400.
After 20 epochs, accuracy on test images is 71.74%, with best validation loss of 0.53,validation accuracy 84.25% and train accuracy
93.88%.
