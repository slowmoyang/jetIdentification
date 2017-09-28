from __future__ import division
from __future__ import print_function

import torch


def choose_optim(args, model_param):
    # Namespace -> dict
    args = vars(args)

    optimizer = torch.optim.__dict__[args['optim']]
    optim_args = create_dict(args['optim'])
    for


    return optimizer(model_param, **optim_args)
