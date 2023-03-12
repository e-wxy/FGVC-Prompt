import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from timm.utils import accuracy, AverageMeter
import os



class Trainer(object):
    def __init__(self, cfg, logger):
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.DEVICE.LOCAL_RANK
        self.prec = cfg.TRAIN.PREC
        self.scaler = GradScaler() if self.prec == "amp" else None
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.stages = []


    def train_one(self, save_name: str, model, train_loader, valid_loader, criterion, optimizer, scheduler, train_cfg):
        """ Training Stage One: contrastive learning

        Args:
            save_name (str): file name for saving
            info (str): info about this training stage to show in logger
            scheduler (torch.optim.lr_scheduler): step scheduler, updated in every batch
            train_cfg (cfg): training configuration for this stage
        """
        # check_period = train_cfg.CHECKPOINT_PERIOD
        log_period = train_cfg.LOG_PERIOD
        eval_period = train_cfg.EVAL_PERIOD
        self.stages.append(train_cfg.MAX_EPOCHS)
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        best_metric = 1e6


        self.logger.info("============ Stage ONE Start [{} Epochs] ============".format(self.stages[-1]))
        model.train()
        step = 0

        for epoch in range(self.stages[-1]):
            loss_meter.reset()
            norm_meter.reset()

            for image, text, _ in train_loader:
                image, text = image.to(self.device), text.to(self.device)
                optimizer.zero_grad()
                if self.prec == "amp":
                    with autocast(enabled=True):
                        sim_g, sim_v, sim_t = model(image, text)
                        loss = criterion(sim_g, sim_v, sim_t)
                    self.scaler.scale(loss).backward()
                    # clip gradient
                    if self.cfg.TRAIN.CLIP_GRAD:
                        self.scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.TRAIN.CLIP_GRAD)
                    else:
                        self.scaler.unscale_(optimizer)
                        grad_norm = get_grad_norm(model.parameters())
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    sim_g, sim_v, sim_t = model(image, text)
                    loss = criterion(sim_g, sim_v, sim_t)
                    loss.backward()
                    if self.cfg.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                    optimizer.step()

                step += 1
                scheduler.step(step)

                torch.cuda.synchronize()
                loss = reduce_tensor(loss)
                grad_norm = reduce_tensor(grad_norm)
                loss_meter.update(loss.item(), image.size(0))
                norm_meter.update(grad_norm)


            self.train_loss.append(loss_meter.avg)

            if epoch % log_period == 0:
                self.logger.info(f'Epoch {epoch+1:3d}: train_loss: {loss_meter.avg:.5f} grad_norm: {norm_meter.avg:.4f}')

            if epoch % eval_period == 0:
                test_loss = self.cal_loss(model, valid_loader, criterion)
                self.test_loss.append(test_loss)
                self.logger.info("Epoch {:3d}: test_loss: {:.5f}".format(epoch+1, test_loss))
                if test_loss < best_metric:
                    if self.device == 0:
                        self.save_state_dict(model, "{}.pt".format("best_1"))
                    best_metric = test_loss
                model.train()

        test_loss = self.cal_loss(model, valid_loader, criterion)
        self.logger.info("Training Stage ONE End: train_loss: {:.5f} test_loss: {:.5f} best_loss: {:.5f}".format(loss_meter.avg, test_loss, best_metric))
        if self.device == 0:
            self.save_state_dict(model, "{}.pt".format(save_name))

    def train_two(self, save_name: str, model, train_loader, valid_loader, criterion, optimizer, scheduler, train_cfg):
        """ Training Stage Two: fine-tune for the classification task

        Args:
            save_name (str): file name for saving
            info (str): info about this training stage to show in logger
            scheduler (torch.optim.lr_scheduler): step scheduler, updated in every batch
            train_cfg (cfg): training configuration for this stage
        """
        # check_period = train_cfg.CHECKPOINT_PERIOD
        log_period = train_cfg.LOG_PERIOD
        eval_period = train_cfg.EVAL_PERIOD
        self.stages.append(train_cfg.MAX_EPOCHS)
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        norm_meter = AverageMeter()
        best_metric = 0


        self.logger.info("============ Stage TWO Start [{} Epochs] ============".format(self.stages[-1]))
        model.train()
        step = 0

        for epoch in range(self.stages[-1]):
            loss_meter.reset()
            acc_meter.reset()
            norm_meter.reset()

            for image, text, label in train_loader:
                image, text, label = image.to(self.device), text.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                if self.prec == "amp":
                    with autocast(enabled=True):
                        z = model(image, text)
                        loss = criterion(z, label)
                    self.scaler.scale(loss).backward()
                    # clip gradient
                    if self.cfg.TRAIN.CLIP_GRAD:
                        self.scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.TRAIN.CLIP_GRAD)
                    else:
                        self.scaler.unscale_(optimizer)
                        grad_norm = get_grad_norm(model.parameters())
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    z = model(image, text)
                    loss = criterion(z, label)
                    loss.backward()
                    if self.cfg.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                    optimizer.step()

                acc = accuracy(z.data, label)[0]
                step += 1
                scheduler.step(step)

                torch.cuda.synchronize()
                loss = reduce_tensor(loss)
                acc = reduce_tensor(acc)
                grad_norm = reduce_tensor(grad_norm)
                loss_meter.update(loss.item(), label.size(0))
                acc_meter.update(acc.item(), label.size(0))
                norm_meter.update(grad_norm)


            self.train_loss.append(loss_meter.avg)
            self.train_acc.append(acc_meter.avg)

            if epoch % log_period == 0:
                self.logger.info("Epoch {:3d}: loss: {:.5f} train_acc: {:.3f}% grad_norm: {:.4f}".format(epoch+1, loss_meter.avg, acc_meter.avg, norm_meter.avg))

            if epoch % eval_period == 0:
                test_acc = self.eval(model, valid_loader)
                self.test_acc.append(test_acc)
                self.logger.info("Epoch {:3d}: test_acc: {:.3f}%".format(epoch+1, test_acc))
                if test_acc > best_metric:
                    if self.device == 0:
                        self.save_state_dict(model, "{}.pt".format("best_2"))
                    best_metric = test_acc
                model.train()

        test_acc = self.eval(model, valid_loader)
        self.logger.info("Training Stage TWO End: train_loss: {:.5f} train_acc: {:.3f}% test_acc: {:.3f}% best_acc: {:.3f}%".format(loss_meter.avg, acc_meter.avg, test_acc, best_metric))
        if self.device == 0:
            self.save_state_dict(model, "{}.pt".format(save_name))


    @torch.no_grad()
    def eval(self, model, val_loader):
        model.eval()
        acc_meter = AverageMeter()

        for image, text, label in val_loader:
            image, text, label = image.to(self.device), text.to(self.device), label.to(self.device)
            # if self.prec == "amp":
            #     with autocast():
            #         z = model(image, text)
            z = model(image, text)
            acc = accuracy(z.data, label)[0]
            torch.cuda.synchronize()
            acc = reduce_tensor(acc)
            acc_meter.update(acc.item(), label.size(0))

        return acc_meter.avg

    @torch.no_grad()
    def cal_loss(self, model, val_loader, criterion):
        model.eval()
        loss_meter = AverageMeter()

        for image, text, _ in val_loader:
            image, text = image.to(self.device), text.to(self.device)
            sim_g, sim_v, sim_t = model(image, text)
            loss = criterion(sim_g, sim_v, sim_t)
            torch.cuda.synchronize()
            loss = reduce_tensor(loss)
            loss_meter.update(loss.item(), image.size(0))

        return loss_meter.avg


    def save_model(self, model, name):
        file_path = os.path.join(self.cfg.OUTPUT_DIR, 'model')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, name)
        if self.cfg.DEVICE.DIST == True:
            model = model.module
        torch.save(model, file_path)
    
    def save_state_dict(self, model, name):
        file_path = os.path.join(self.cfg.OUTPUT_DIR, 'model')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, name)
        if self.cfg.DEVICE.DIST == True:
            model = model.module
        torch.save(model.state_dict(), file_path)

    def record_training_process(self):
        self.logger.info("============ Training Process Record============")
        self.logger.info("Stages: {}".format(self.stages))
        self.logger.info("\ntrain_loss = {}\ntest_loss = {}\ntrain_acc = {}\ntest_acc = {}".format(self.train_loss, self.test_loss, self.train_acc, self.test_acc))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm