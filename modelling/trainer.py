import numpy as np
import os, sys
import torch
import time
import datetime
from modelling.train_utils import loss_maker, accuracy_maker, prediction_maker, target_maker, map_maker
from toolbox.save_utils import save_mat, save_cam_map
from toolbox.plot_utils import augment_images


class Trainer:
    def __init__(self, net, optimizer, schedule, recorder, train_dataloader, test_dataloader, config, args, device):
        self.net = net
        self.optimizer = optimizer
        self.schedule = schedule
        self.recorder = recorder
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.config = config
        self.args = args
        self.device = device
        print(self.args)

    def train(self):
        for epoch_no in range(self.args.num_epochs):
            self.net.train(mode=True)
            self.train_epoch(epoch_no)
            self.val_epoch(epoch_no)
            self.schedule.step()
            self.recorder.cat_info(epoch_no)
            print('Current learning rate: {}'.format(self.schedule.get_last_lr()))

            disp_epoch_no = epoch_no + 1
            if np.mod(disp_epoch_no, self.args.save_interval) == 0 or disp_epoch_no == self.args.num_epochs:
                torch.save(self.net.module.state_dict(),
                           os.path.join(self.args.save_path, 'net_e{}.ckpt'.format(disp_epoch_no)))
                self.recorder.plot(path=self.args.save_path)
                save_mat(self.args.save_path, self.recorder.master_dict)

        return self.recorder

    def train_epoch(self, epoch_no):

        disp_epoch_no = epoch_no + 1
        num_batches = int(np.floor(len(self.train_loader.dataset) / self.train_loader.batch_size))

        disp_values = 0
        num_processed = 0
        start_time = time.time()
        for batch_no, (source_images, images, targets, masks) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            disp_batch_no = batch_no + 1
            processed_batch_size = images.shape[0]

            images = images.to(self.device)
            targets = targets.float().to(self.device)
            attn_targets = masks.to(self.device)

            cls_logits, attn_logits, cls_maps = self.net(images)
            cls_targets, attn_targets = target_maker(targets, attn_targets, self.config)

            losses = loss_maker(cls_logits, attn_logits, cls_targets, attn_targets, self.config)
            accus, num_valid_cases = accuracy_maker(cls_logits, attn_logits, cls_targets, attn_targets, self.config)
            preds, probs, targets = prediction_maker(cls_logits, cls_targets, self.config)

            self.recorder.add_info(epoch_no, 'tra', losses)
            self.recorder.add_info(epoch_no, 'tra', accus)
            self.recorder.add_info(epoch_no, 'tra', num_valid_cases)
            self.recorder.add_info(epoch_no, 'tra', preds)
            self.recorder.add_info(epoch_no, 'tra', probs)
            self.recorder.add_info(epoch_no, 'tra', targets)
            self.recorder.add_info(epoch_no, 'tra', {'batch_size': [processed_batch_size]})

            train_loss = torch.stack(list(losses.values())).sum()
            disp_values += np.array([train_loss.item() * processed_batch_size] +
                                    [v.item() * n.item() for v, n in zip(losses.values(), num_valid_cases.values())] +
                                    [v.item() * n.item() for v, n in zip(accus.values(), num_valid_cases.values())])
            disp_names = ['loss'] + list(losses.keys()) + list(accus.keys())

            num_processed += np.array([processed_batch_size] + [n.item() for n in num_valid_cases.values()] * 2)
            avg_disp_values = dict(zip(disp_names, (disp_values / np.maximum(num_processed, 1)).tolist()))

            # back propagation and optimize
            train_loss.backward()
            self.optimizer.step()

            # add time printing
            elapsed_time = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if np.mod(disp_batch_no, self.args.display_interval) == 0 or disp_batch_no == num_batches:
                disp_str_tmp = '[{} Elapsed: {}] [tra] [epoch{} {}/{}] '
                disp_str_tmp += ' '.join(['{}: {:.3f}'.format(s, avg_disp_values[s]) for s in avg_disp_values.keys()])
                print(
                    disp_str_tmp.format(
                        current_time, elapsed_time,
                        disp_epoch_no, disp_batch_no, num_batches,
                    ))

                # break
        return 0

    def val_epoch(self, epoch_no):
        self.net.eval()

        disp_epoch_no = epoch_no + 1
        num_batches = int(np.ceil(len(self.test_loader.dataset) / self.test_loader.batch_size))

        disp_values = 0
        num_processed = 0
        augmented_maps = dict()
        start_time = time.time()
        with torch.no_grad():
            for batch_no, (source_images, images, targets, masks) in enumerate(self.test_loader):
                disp_batch_no = batch_no + 1
                processed_batch_size = images.shape[0]

                images = images.to(self.device)
                targets = targets.float().to(self.device)
                attn_targets = masks.to(self.device)

                cls_logits, attn_logits, cls_maps = self.net(images)

                cls_targets, attn_targets = target_maker(targets, attn_targets, self.config)
                losses = loss_maker(cls_logits, attn_logits, cls_targets, attn_targets, self.config)
                accus, num_valid_cases = accuracy_maker(cls_logits, attn_logits, cls_targets, attn_targets, self.config)
                preds, probs, targets = prediction_maker(cls_logits, cls_targets, self.config)

                if (np.mod(disp_epoch_no, self.args.save_interval) == 0 or disp_epoch_no == self.args.num_epochs) \
                        and np.mod(disp_batch_no, self.args.cam_interval) == 0 \
                        and self.args.cam_interval != -1:
                    maps = map_maker(cls_maps, self.config)

                self.recorder.add_info(epoch_no, 'val', losses)
                self.recorder.add_info(epoch_no, 'val', accus)
                self.recorder.add_info(epoch_no, 'val', num_valid_cases)
                self.recorder.add_info(epoch_no, 'val', preds)
                self.recorder.add_info(epoch_no, 'val', probs)
                self.recorder.add_info(epoch_no, 'val', targets)
                self.recorder.add_info(epoch_no, 'val', {'batch_size': [processed_batch_size]})

                val_loss = torch.stack(list(losses.values())).sum()
                disp_values += np.array([val_loss.item() * processed_batch_size] +
                                        [v.item() * n.item() for v, n in
                                         zip(losses.values(), num_valid_cases.values())] +
                                        [v.item() * n.item() for v, n in zip(accus.values(), num_valid_cases.values())])
                disp_names = ['loss'] + list(losses.keys()) + list(accus.keys())

                num_processed += np.array([processed_batch_size] + [n.item() for n in num_valid_cases.values()] * 2)
                avg_disp_values = dict(zip(disp_names, (disp_values / np.maximum(num_processed, 1)).tolist()))

                if (np.mod(disp_epoch_no, self.args.save_interval) == 0 or disp_epoch_no == self.args.num_epochs) \
                        and np.mod(disp_batch_no, self.args.cam_interval) == 0 \
                        and self.args.cam_interval != -1:
                    for m, t, p, pp in zip(maps, targets, preds, probs):
                        if m not in augmented_maps:
                            augmented_maps[m] = list()
                        augmented_maps[m].extend(
                            augment_images(source_images, maps[m], targets[t], preds[p], probs[pp], config=self.config))

                # add time printing
                elapsed_time = str(datetime.timedelta(seconds=round(time.time() - start_time)))
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if np.mod(disp_batch_no, self.args.display_interval) == 0 or disp_batch_no == num_batches:
                    disp_str_tmp = '[{} Elapsed: {}] [val] [epoch{} {}/{}] '
                    disp_str_tmp += ' '.join(
                        ['{}: {:.3f}'.format(s, avg_disp_values[s]) for s in avg_disp_values.keys()])
                    print(
                        disp_str_tmp.format(
                            current_time, elapsed_time,
                            disp_epoch_no, disp_batch_no, num_batches,
                        )
                    )

        if np.mod(disp_epoch_no, self.args.save_interval) == 0 or disp_epoch_no == self.args.num_epochs:
            for m in augmented_maps:
                save_cam_map(self.args.save_path, augmented_maps[m], m, disp_epoch_no)

        return 0
