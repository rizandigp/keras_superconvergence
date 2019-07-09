from keras.callbacks import Callback, LambdaCallback
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import math

class CyclicalLR(Callback):
    '''Cyclical learning rate [Leslie Smith 2015] callback with custom policies and momentum.

    "Instead of monotonically decreasing the learning rate, this method lets the learning rate cyclically vary 
    between reasonable boundary values. Training with cyclical learning rates instead of fixed values achieves 
    improved classification accuracy without a need to tune and often in fewer iterations."
    
    Arguments:
        cycle_fn (callable): Function depicting the cycle, taking in a float (0 ~ inf) as the phase in the cycle,
            returning the fraction (0 ~ 1) of max_lr for that phase in the cycle. Example:
            >>> def triangular(x):
            >>> x,_ = math.modf(x)
            >>> if x<0.5:
            >>>     return x*2.0
            >>> else:
            >>>     return 1 - x*2.0
        cycle_mom_fn (callable): Function depicting the momentum part of the cycle. If None, defaults to 1-cycle_fn(x)
        steps_per_epoch (int): Should be ceil(num_samples/batch_size)
        max_lr (float): Maximum learning rate. Should be found out using LRFinder
        num_cycles (int): A good value would be between (1~4)*num_epochs according to [1] or 1 for super-convergence according to [2]
        div_factor (float): Lowest learning rate in the cycle will be max_lr/div_factor.
        max_momentum (float): Optimal value can be found out with a grid search (e.g. 90, 95, 99) [4]
        min_momentum (float): According to [4], experiments show this doesn't really matter, and 0.85 is a good default. Though still worth experimenting with.

    References:
        [1] https://arxiv.org/pdf/1506.01186.pdf 
        [2] https://arxiv.org/pdf/1708.07120.pdf
        [3] https://arxiv.org/pdf/1608.03983.pdf
        [4] https://towardsdatascience.com/https-medium-com-super-convergence-very-fast-training-of-neural-networks-using-large-learning-rates-decb689b9eb0 
        '''
    def __init__(self, cycle_fn, max_lr, num_cycles, num_epochs, steps_per_epoch, cycle_mom_fn=None, div_factor=25.0, 
                max_momentum=0.95, min_momentum=0.85):
        super(CyclicalLR, self).__init__()
        self.cycle_fn = cycle_fn
        self.cycle_mom_fn = cycle_mom_fn
        self.max_lr = max_lr
        self.min_lr = max_lr/div_factor
        self.num_cycles = num_cycles
        self.num_epochs = num_epochs
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.iterations_per_cycle = steps_per_epoch*num_epochs/num_cycles

        self.iteration = 0

    def on_batch_begin(self, epoch, logs):
        x = self.iteration/self.iterations_per_cycle # Cycle phase

        lr = self.min_lr + (self.max_lr - self.min_lr) * self.cycle_fn(x)
        K.set_value(self.model.optimizer.lr, lr)

        if hasattr(self.model.optimizer, 'momentum'):
            if self.cycle_mom_fn==None:
                mom_fraction = 1.0 - self.cycle_fn(x)
            else:
                mom_fraction = self.cycle_mom_fn(x)
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * mom_fraction
            K.set_value(self.model.optimizer.momentum, momentum)

        self.iteration += 1


class LRFinder():
    '''Learning rate finder for cyclical learning rates a la fastai.
    '''
    def __init__(self, model, copy_model=True, custom_objects={}):
        # Copy the model
        # Known issue: when using tf backend, there seems to be a memory leak
        # that causes saving and loading models to get slower with each successive call.
        # You can work around this by clearing the session: K.clear_session()
        if copy_model:
            model.save('tmp.h5')
            self.model = load_model('tmp.h5', custom_objects=custom_objects)
            os.remove('tmp.h5')
        else:
            self.model = model

        self.lr = []
        self.loss = []
        self.max_loss = None

    def find(self, X, y, start_lr, end_lr, batch_size, num_epochs=1, max_loss=None, **kwargs):
        steps_per_epoch = X.shape[0] / batch_size
        self.lr_multiplier = (end_lr / start_lr) ** (1.0 / steps_per_epoch / num_epochs)
        self.max_loss = max_loss

        K.set_value(self.model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=self.on_batch_end)
        self.model.fit(X, y, batch_size=batch_size, epochs=num_epochs, callbacks=[callback], **kwargs)

    def find_generator(self, generator, start_lr, end_lr, steps_per_epoch=None, num_epochs=1, max_loss=None, **kwargs):
        if steps_per_epoch==None:
            steps_per_epoch = len(generator)
        self.lr_multiplier = (end_lr / start_lr) ** (1.0 / float(steps_per_epoch/num_epochs))
        self.max_loss = max_loss

        K.set_value(self.model.optimizer.lr, start_lr)
        callback = LambdaCallback(on_batch_end=self.on_batch_end)
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, callbacks=[callback], **kwargs)

    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        loss = logs['loss']
        self.lr.append(lr)
        self.loss.append(loss)

        if math.isnan(loss) or (self.max_loss is not None and loss > self.max_loss):
            self.model.stop_training = True
            return

        lr *= self.lr_multiplier
        K.set_value(self.model.optimizer.lr, lr)

    def plot_losses(self, skip_start=0, skip_end=0, **kwargs):
        plt.plot(self.lr[skip_start:-(skip_end+1)], self.loss[skip_start:-(skip_end+1)], **kwargs)
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate")
        plt.xscale('log')