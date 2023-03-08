from .model import TokenCLIP, SimCLIP, ClsCLIP, BaseCLIP, VisualCLIP


def build_training_model(cfg, type="tokenflow"):
    if type == "basic":
        model = BaseCLIP(cfg)
    else:
        model = SimCLIP(cfg)
    return model

def build_cls_model(cfg, clip_model, type="clip"):
    if type == "visual":
        model = VisualCLIP(cfg, clip_model)
    else:
        model = ClsCLIP(cfg, clip_model)
    return model