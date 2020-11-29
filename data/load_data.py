import tensorflow as tf
import tensorflow_datasets as tfd

ds = tfd.load('clevr', split='train', shuffle_files=True)
print(ds)

if __name__ == '__main__':
    pass
