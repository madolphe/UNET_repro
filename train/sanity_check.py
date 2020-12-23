import matplotlib.pyplot as plt

class Sanity_Check:
    def __init__(self, ds, model, training):
        self.ds = ds
        self.model = model
        self.training = training

    def check_output_model(self):
        img = next(iter(self.ds))
        print(img.keys())
        print(img['image'].shape)
        print(img['mask'].shape)
        output = self.model(img['image'])
        print(output.shape)
        plt.imshow(output[0,:,:,0])
        plt.show()
    def check_CE_init():
        pass
