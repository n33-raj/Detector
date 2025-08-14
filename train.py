#train.py

import argparse
import copy
import csv
import os
import warnings

import numpy as np
import torch
import tqdm
from utils import util
import yaml
import torch.nn as nn
from model.model import yolo_v8_n
import dataloader
from pathlib import Path


import os
from glob import glob
from torch.utils.data import DataLoader

from model import *
from dataloader.dataloader import Dataset

warnings.filterwarnings("ignore")


FORMATS = ('bmp','dng','jpeg','jpg','mpo','png','tif','tiff','webp')

def _list_images(root):
    root = Path(root)
    out = []
    for ext in FORMATS:
        out += [str(p) for p in root.rglob(f'*.{ext}')]
        out += [str(p) for p in root.rglob(f'*.{ext.upper()}')]
    return out


class YoloTrainer:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = yolo_v8_n(len(params['names'].values())).to(self.device)

        self.accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
        self.params['weight_decay'] *= args.batch_size * args.world_size * self.accumulate / 64

        self.optimizer = self._setup_optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=self._get_lr_lambda()
        )
        self.ema = util.EMA(self.model) if args.local_rank == 0 else None
        self.criterion = util.ComputeLoss(self.model, self.params)
        self.scaler = torch.cuda.amp.GradScaler()

        self.best_map = 0.0

        if args.local_rank == 0:
            os.makedirs('weights', exist_ok=True)

    def _get_lr_lambda(self):
        """Returns the learning rate lambda function"""
        def lr_lambda(epoch):
            return (1 - epoch / self.args.epochs) * (1.0 - self.params['lrf']) + self.params['lrf']
        return lr_lambda

    def _setup_optimizer(self):
        biases, bn_weights, weights = [], [], []
        for module in self.model.modules():
            if hasattr(module, 'bias') and isinstance(module.bias, torch.nn.Parameter):
                biases.append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_weights.append(module.weight)
            elif hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                weights.append(module.weight)

        optimizer = torch.optim.SGD(biases, lr=self.params['lr0'], momentum=self.params['momentum'], nesterov=True)
        optimizer.add_param_group({'params': weights, 'weight_decay': self.params['weight_decay']})
        optimizer.add_param_group({'params': bn_weights})

        return optimizer

    def _load_dataset(self, train=True):
        # Prefer reading from config.yml:
        # Expect keys like params['data']['train'] and params['data']['val'], or similar.
        data_cfg = self.params.get('data', {})
        img_dir = data_cfg.get('train' if train else 'val', None)

        # Fallback: try YOLO-style default relative layout if config doesn't provide paths
        if img_dir is None:
            base = Path('dataset') / 'images'
            img_dir = str(base / ('train' if train else 'val'))

        label_root = Path(img_dir.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'))

        # Collect images
        image_paths_all = _list_images(img_dir)
        image_paths = []
        for img_path in image_paths_all:
            stem = Path(img_path).stem
            label_path = label_root / f'{stem}.txt'
            if label_path.exists():
                image_paths.append(img_path)

        if len(image_paths) == 0:
            raise ValueError(
                f'No images found.\n'
                f'  Searched: {img_dir}\n'
                f'  Did you set correct paths in config.yml under data.train / data.val?\n'
                f'  Also ensure labels live under ".../labels/<split>/*.txt" and names match images.'
            )

        print(f"Loaded {len(image_paths)} {'training' if train else 'validation'} samples")
        return Dataset(image_paths, self.args.input_size, self.params, augment=train)
    
    def train(self):
        dataset = self._load_dataset(train=True)

        sampler = None if self.args.world_size <= 1 else data.distributed.DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )

        if self.args.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )

        num_batches = len(loader)
        num_warmup_iters = max(round(self.params['warmup_epochs'] * num_batches), 1000)

        with open('weights/step.csv', 'w', newline='') as f_csv:
            if self.args.local_rank == 0:
                writer = csv.DictWriter(f_csv, fieldnames=['epoch', 'mAP@50', 'mAP'])
                writer.writeheader()

            for epoch in range(self.args.epochs):
                self.model.train()

                if self.args.epochs - epoch <= 10:
                    loader.dataset.mosaic = False

                if self.args.world_size > 1:
                    sampler.set_epoch(epoch)

                progress_bar = tqdm.tqdm(enumerate(loader), total=num_batches) if self.args.local_rank == 0 else enumerate(loader)

                running_loss = util.AverageMeter()
                self.optimizer.zero_grad()

                for i, (images, targets, _) in progress_bar:
                    iteration = i + epoch * num_batches
                    images = images.to(self.device).float() / 255.0
                    targets = targets.to(self.device)

                    # Warmup
                    if iteration <= num_warmup_iters:
                        warmup_ratio = np.interp(iteration, [0, num_warmup_iters], 
                                               [1, 64 / (self.args.batch_size * self.args.world_size)])
                        self.accumulate = max(1, round(warmup_ratio))
                        
                        for j, group in enumerate(self.optimizer.param_groups):
                            if j == 0:  # biases
                                group_lr = np.interp(iteration, [0, num_warmup_iters],
                                                   [self.params['warmup_bias_lr'],
                                                   group['initial_lr'] * self._get_lr_lambda()(epoch)])
                            else:  # weights
                                group_lr = np.interp(iteration, [0, num_warmup_iters],
                                                   [0.0,
                                                   group['initial_lr'] * self._get_lr_lambda()(epoch)])
                            
                            group['lr'] = group_lr
                            if 'momentum' in group:
                                group['momentum'] = np.interp(iteration, [0, num_warmup_iters],
                                                            [self.params['warmup_momentum'], 
                                                            self.params['momentum']])

                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                    running_loss.update(loss.item(), images.size(0))
                    loss = loss * self.args.batch_size * self.args.world_size

                    self.scaler.scale(loss).backward()

                    if iteration % self.accumulate == 0:
                        self.scaler.unscale_(self.optimizer)
                        util.clip_gradients(self.model)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.ema:
                            self.ema.update(self.model)

                    if self.args.local_rank == 0:
                        mem = torch.cuda.memory_reserved() / 1e9
                        progress_bar.set_description(f'Epoch [{epoch+1}/{self.args.epochs}] Memory: {mem:.3f}G Loss: {running_loss.avg:.4g}')

                self.scheduler.step()

                if self.args.local_rank == 0:
                    map50, mean_ap = self.test(model=self.ema.ema if self.ema else self.model)
                    writer.writerow({'epoch': str(epoch + 1).zfill(3), 'mAP@50': f'{map50:.3f}', 'mAP': f'{mean_ap:.3f}'})
                    f_csv.flush()

                    if mean_ap > self.best_map:
                        self.best_map = mean_ap
                        ckpt = {'model': copy.deepcopy(self.ema.ema if self.ema else self.model).half()}
                        torch.save(ckpt, 'weights/best.pt')
                        del ckpt
                    ckpt = {'model': copy.deepcopy(self.ema.ema if self.ema else self.model).half()}
                    torch.save(ckpt, 'weights/last.pt')
                    del ckpt

        if self.args.local_rank == 0:
            util.strip_optimizer('weights/best.pt')
            util.strip_optimizer('weights/last.pt')

        torch.cuda.empty_cache()

    @torch.no_grad()
    def test(self, model=None):
        dataset = self._load_dataset(train=False)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )

        if model is None:
            ckpt = torch.load('weights/best.pt', map_location=self.device)
            model = ckpt['model'].to(self.device).float()

        model.eval()
        model.half()

        iou_thresholds = torch.linspace(0.5, 0.95, 10).to(self.device)
        n_iou = iou_thresholds.numel()

        metrics = []
        pbar = tqdm.tqdm(loader, desc='Evaluating')

        for images, targets, shapes in pbar:
            images = images.to(self.device).half() / 255.0
            targets = targets.to(self.device)
            batch_size, _, height, width = images.shape

            outputs = model(images)
            targets[:, 2:] *= torch.tensor([width, height, width, height], device=self.device)

            outputs = util.non_max_suppression(outputs, conf_thres=0.001, iou_thres=0.65)

            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=self.device)

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), labels[:, 0]))
                    continue

                detections = output.clone()
                util.scale(detections[:, :4], images[i].shape[1:], shapes[i][0], shapes[i][1])

                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()
                    tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2
                    tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2
                    tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2
                    tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2
                    util.scale(tbox, images[i].shape[1:], shapes[i][0], shapes[i][1])

                    correct_np = np.zeros((detections.shape[0], n_iou), dtype=bool)
                    t_tensor = torch.cat((labels[:, 0:1], tbox), dim=1)
                    iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]

                    for j in range(n_iou):
                        matches = torch.where((iou >= iou_thresholds[j]) & correct_class)
                        if matches[0].shape[0]:
                            matched = torch.cat((torch.stack(matches, dim=1), iou[matches[0], matches[1]].unsqueeze(1)), dim=1).cpu().numpy()
                            if matched.shape[0] > 1:
                                matched = matched[matched[:, 2].argsort()[::-1]]
                                matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
                                matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
                            correct_np[matched[:, 1].astype(int), j] = True

                    correct = torch.tensor(correct_np, dtype=torch.bool, device=self.device)

                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        if metrics:
            metrics_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
            if len(metrics_np) and metrics_np[0].any():
                tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics_np)
            else:
                m_pre = m_rec = map50 = mean_ap = 0.0
        else:
            m_pre = m_rec = map50 = mean_ap = 0.0

        print(f'Precision: {m_pre:.3g} Recall: {m_rec:.3g} mAP: {mean_ap:.3g}')
        model.float()
        return map50, mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=51000)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        os.makedirs('weights', exist_ok=True)

    util.setup_seed()
    util.setup_multi_processes()

    # with open('config.yml', errors='ignore') as f:
    #     params = yaml.safe_load(f)


    with open(os.path.join('config', 'config.yml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    trainer = YoloTrainer(args, params)

    if args.train:
        trainer.train()
    if args.test:
        trainer.test()


if __name__ == '__main__':
    main()