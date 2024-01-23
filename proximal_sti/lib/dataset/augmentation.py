import torchio as tio

def get_transform():
    training_transform = tio.Compose(
        [
            tio.OneOf({                                # either
                tio.RandomAffine(default_pad_value=0): 0.8,               # random affine
                tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
            }, p=0.8), 
        ]
    )
    return training_transform
