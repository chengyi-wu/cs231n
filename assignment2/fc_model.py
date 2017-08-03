from __future__ import print_function
import time
import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

best_model = None
best_val_acc = -1
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

num_train = 100
input_dim = 3072
num_classes = 10

learning_rates = np.random.uniform(low=1e-4, high=1e-1,size=10000)
weight_scales = np.random.uniform(low=2e-4, high=1e-1,size=10000)
dropouts = np.random.uniform(low=0, high=1, size=10)

for i in range(num_train):
    
    use_bn = True
    reg = 2.5
    lr = np.random.choice(learning_rates, 1)[0]
    weight_scale = np.random.choice(weight_scales, 1)[0]
    dropout = np.random.choice(dropouts, 1)[0]
    
    print(lr, weight_scale, dropout)

    model = FullyConnectedNet([100, 100, 100, 100],
                              input_dim=input_dim,
                              num_classes=num_classes,
                              dropout=dropout,
                              use_batchnorm=use_bn,
                              reg=reg,
                              weight_scale=weight_scale, 
                              dtype=np.float64)
    solver = Solver(model, data,
                    num_epochs=25, batch_size=100,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': lr,
                    },
                    lr_decay=0.95,
                    verbose=False, print_every=100
             )
    solver.train()

    if solver.best_val_acc > best_val_acc:
        best_val_acc = solver.best_val_acc
        print('best_val_acc: ', best_val_acc)

################################################################################
#                              END OF YOUR CODE                                #
################################################################################