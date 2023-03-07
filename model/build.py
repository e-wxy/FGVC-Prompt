from .model import TokenCLIP, SimCLIP, ClsCLIP, BaseCLIP


def build_training_model(cfg):
    # model = SimCLIP(cfg)
    model = BaseCLIP(cfg)
    return model

def build_cls_model(cfg, clip_model):
    model = ClsCLIP(cfg, clip_model)
    return model