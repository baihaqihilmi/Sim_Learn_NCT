from torch import optim


def get_optimizer(cfg, params):
    if cfg.OPTIMIZER == "Adam":
        optimizer = optim.Adam(params ,  lr=cfg.LEARNING_RATE)
    elif cfg.OPTIMIZER == "SGD":
        optimizer = optim.SGD(params ,  lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(params ,  lr=cfg.LEARNING_RATE)
    else:
        raise NotImplementedError(f"Optimizer {cfg.OPTIMIZER} not implemented")

    return optimizer