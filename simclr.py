import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import (
    save_config_file,
    accuracy,
    save_checkpoint,
    save_json,
    save_metrics_csv,
    load_metrics_csv,
    save_training_curves,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # pragma: no cover - simple runtime fallback
        def __init__(self, log_dir=None):
            self.log_dir = log_dir

        def add_scalar(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.run_dir)
        self.log_path = os.path.join(self.writer.log_dir, 'training.log')
        os.makedirs(self.writer.log_dir, exist_ok=True)
        logging.basicConfig(filename=self.log_path, level=logging.DEBUG, force=True)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, start_epoch=0):
        start_time = time.time()
        scaler = GradScaler(enabled=self.args.fp16_precision)
        metrics_csv_path = os.path.join(self.writer.log_dir, 'metrics.csv')
        existing_metrics = load_metrics_csv(metrics_csv_path) if start_epoch > 0 else []
        epoch_metrics = [row for row in existing_metrics if row['epoch'] <= start_epoch]
        saved_checkpoints = []
        if start_epoch > 0 and self.args.checkpoint_every_n_epochs > 0:
            for checkpoint_epoch in range(self.args.checkpoint_every_n_epochs, start_epoch + 1, self.args.checkpoint_every_n_epochs):
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(checkpoint_epoch)
                checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)
                if os.path.exists(checkpoint_path):
                    saved_checkpoints.append({
                        'epoch': checkpoint_epoch,
                        'checkpoint_path': checkpoint_path,
                    })

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        if epoch_metrics:
            n_iter = len(epoch_metrics) * len(train_loader)
            last_top1 = epoch_metrics[-1]['top1']
            last_top5 = epoch_metrics[-1]['top5']
            last_loss = epoch_metrics[-1]['loss']
        else:
            last_top1 = 0.0
            last_top5 = 0.0
            last_loss = 0.0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        if start_epoch > 0:
            logging.info(f"Resuming training from epoch {start_epoch}.")

        for epoch_counter in range(start_epoch, self.args.epochs):
            epoch_loss_total = 0.0
            epoch_top1_total = 0.0
            epoch_top5_total = 0.0
            epoch_steps = 0
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                loss_value = float(loss.item())
                top1_value = float(top1[0].item())
                top5_value = float(top5[0].item())

                epoch_loss_total += loss_value
                epoch_top1_total += top1_value
                epoch_top5_total += top5_value
                epoch_steps += 1

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss_value, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1_value, global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5_value, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            last_loss = epoch_loss_total / max(epoch_steps, 1)
            last_top1 = epoch_top1_total / max(epoch_steps, 1)
            last_top5 = epoch_top5_total / max(epoch_steps, 1)
            current_lr = float(self.optimizer.param_groups[0]['lr'])

            epoch_summary = {
                'epoch': epoch_counter + 1,
                'loss': round(last_loss, 6),
                'top1': round(last_top1, 4),
                'top5': round(last_top5, 4),
                'learning_rate': current_lr,
            }
            epoch_metrics.append(epoch_summary)
            logging.debug(
                f"Epoch: {epoch_counter + 1}\tLoss: {last_loss:.6f}\tTop1 accuracy: {last_top1:.4f}\tTop5 accuracy: {last_top5:.4f}"
            )

            if self.args.checkpoint_every_n_epochs > 0:
                should_save = (
                    (epoch_counter + 1) % self.args.checkpoint_every_n_epochs == 0
                    and (epoch_counter + 1) != self.args.epochs
                )
                if should_save:
                    intermediate_checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter + 1)
                    intermediate_checkpoint_path = os.path.join(self.writer.log_dir, intermediate_checkpoint_name)
                    save_checkpoint({
                        'epoch': epoch_counter + 1,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }, is_best=False, filename=intermediate_checkpoint_path)
                    saved_checkpoints.append({
                        'epoch': epoch_counter + 1,
                        'checkpoint_path': intermediate_checkpoint_path,
                    })
                    logging.info(f"Saved intermediate checkpoint at epoch {epoch_counter + 1}: {intermediate_checkpoint_path}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, is_best=False, filename=checkpoint_path)
        saved_checkpoints = [item for item in saved_checkpoints if item['epoch'] != self.args.epochs]
        saved_checkpoints.append({
            'epoch': self.args.epochs,
            'checkpoint_path': checkpoint_path,
        })
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        save_metrics_csv(epoch_metrics, metrics_csv_path)
        training_curves_path = save_training_curves(
            epoch_metrics,
            os.path.join(self.writer.log_dir, 'training_curves.png'),
        )

        duration_seconds = round(time.time() - start_time, 2)
        summary = {
            'experiment_name': self.args.experiment_name,
            'augmentation': self.args.augmentation,
            'dataset_name': self.args.dataset_name,
            'arch': self.args.arch,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'temperature': self.args.temperature,
            'seed': self.args.seed,
            'device': str(self.args.device),
            'run_dir': self.writer.log_dir,
            'checkpoint_path': checkpoint_path,
            'metrics_csv_path': metrics_csv_path,
            'training_log_path': self.log_path,
            'training_curves_path': training_curves_path,
            'final_loss': round(last_loss, 6),
            'final_top1': round(last_top1, 4),
            'final_top5': round(last_top5, 4),
            'duration_seconds': duration_seconds,
            'saved_checkpoints': saved_checkpoints,
            'epoch_metrics': epoch_metrics,
        }
        save_json(summary, os.path.join(self.writer.log_dir, 'summary.json'))
        return summary
