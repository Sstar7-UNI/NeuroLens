from neurolens.controller import AugmentationConfig


class TrainingConfig:
   def __init__(self, epochs:int,batchSize:int, learningRate:float,
                weightDecay:float, dropoutRate:float,decoupledDecay:bool,
                patience:int,useAugmentation:bool, augmentation:AugmentationConfig, seed:int,
                emaAlpha:float = 0.4, useClassWeights:bool = False):
       self.epochs = epochs
       self.batchSize = batchSize
       self.learningRate = learningRate
       self.weightDecay = weightDecay
       self.dropoutRate = dropoutRate
       self.decoupledDecay = decoupledDecay
       self.patience = patience
       self.useAugmentation = useAugmentation
       self.augmentation = augmentation
       self.seed = seed
       self.emaAlpha = emaAlpha
       self.useClassWeights = useClassWeights