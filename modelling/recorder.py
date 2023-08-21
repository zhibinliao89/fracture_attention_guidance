import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class Recorder:
    def __init__(self):
        self.master_dict = dict()

    def get_info_dict(self, epoch_no, tra_val_tag):

        key = epoch_no_str(epoch_no)
        if key not in self.master_dict:
            self.epoch_dict_factory(key)

        return self.master_dict[key][tra_val_tag]

    def epoch_dict_factory(self, key):
        epoch_dict = {'tra': dict(), 'val': dict()}

        self.master_dict[key] = epoch_dict

    def add_info(self, epoch_no, tra_val_tag, new_info):

        info_dict = self.get_info_dict(epoch_no, tra_val_tag)

        for k in new_info:
            if k not in info_dict.keys():
                info_dict[k] = list()

            v = new_info[k]
            if type(v) == torch.Tensor:
                v = v.cpu().detach().numpy()
                if len(v.shape) >= 2:
                    if np.prod(v.shape)/v.shape[0] > 1000 and np.mod(epoch_no + 1, 5) != 0:
                        v = np.array([[0]], v.dtype)

            info_dict[k].append(v)

    def cat_info(self, epoch_no):

        for tra_val_tag in ['tra', 'val']:
            info_dict = self.get_info_dict(epoch_no, tra_val_tag)
            for key in info_dict:
                info_dict[key] = np.concatenate(info_dict[key], axis=0)

    def plot(self, path, plot_tag=('loss', 'accu')):

        keys = self.master_dict['e1']['tra'].keys()

        for k in keys:
            for p_tag in plot_tag:
                if k[:len(p_tag)] == p_tag:
                    plt.figure(1000)
                    plt.clf()

                    # if p_tag == 'loss':
                    #     plt.axes().set_yscale('log')

                    for tra_val_tag, line_style, color_style in zip(['tra', 'val'], ['-', '--'], ['g', 'r']):
                        # batch_size = [self.master_dict[e][tra_val_tag]['batch_size'] for e in self.master_dict]
                        batch_size = [self.master_dict[e][tra_val_tag]['nvac' + k[len(p_tag):]]
                                      for e in self.master_dict]
                        normalizer = [np.maximum(b.cumsum(), 1.) for b in batch_size]
                        values = [self.master_dict[e][tra_val_tag][k] for e in self.master_dict]
                        values_cumsum = [(v*b).cumsum() for v, b in zip(values, batch_size)]

                        x = [n/n.max() + idx for idx, n in enumerate(normalizer)]
                        y = np.array(values_cumsum)/np.array(normalizer)

                        x = [np.array([v[-1]]) for v in x]
                        y = [np.array([v[-1]]) for v in y]

                        x = np.concatenate(x, axis=0)
                        y = np.concatenate(y, axis=0)

                        plt.plot(x, y, color_style + 'o' + line_style, label=tra_val_tag, markersize=1.)

                    plt.grid()
                    plt.legend()
                    # handles, labels = plt.axes().get_legend_handles_labels()
                    plt.xlabel('#epochs')
                    plt.ylabel(k)
                    # plt.legend(handles, labels)
                    plt.savefig(os.path.join(path, k + '.svg'))
                    plt.close()


def epoch_no_str(epoch_no):
    return 'e' + str(epoch_no + 1)
