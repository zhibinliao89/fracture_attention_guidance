import matplotlib.pyplot as plt
import socket
import matplotlib
import numpy as np
import cv2
import torch.nn.functional as F
import torch

matplotlib.use('Agg')

global_figure_no = 0


def augment_images(source_images, cams, labels, preds, probs, force_resize=False, resize_size=224, config=None):

    if 'num_cam_sample_limit' in config:
        num_cam_sample_limit = config['num_cam_sample_limit']
    else:
        num_cam_sample_limit = 10

    if force_resize:
        source_images = F.interpolate(source_images, [resize_size, resize_size], mode='bicubic', align_corners=True).clamp(0., 1.)

    augmented_images = list()
    for im_idx, sim in enumerate(source_images[:min(num_cam_sample_limit, source_images.shape[0])]):
        # the image
        sim = sim.numpy()
        sim = np.moveaxis(sim, (0, 1, 2), (2, 0, 1))

        # attention map
        att_maps = cams[im_idx].detach().cpu().numpy()
        for cl in range(min(att_maps.shape[0], 4)):
            att_map = att_maps[cl]
            h, w = att_map.shape
            att_map = cv2.resize(att_map, source_images.shape[2:], interpolation=cv2.INTER_CUBIC)

            min_value = np.min(np.min(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
            max_value = np.max(np.max(att_map, axis=0, keepdims=True), axis=1, keepdims=True)
            att_map = (att_map - min_value) / (max_value - min_value)

            # text to display
            l = labels[im_idx].item()
            p = preds[im_idx].item()

            if l.is_integer():
                l = int(l)
                if 'label_short_name' in config:
                    l_text = config['label_short_name'][l]
                else:
                    l_text = f'{l}'
            else:
                l_text = '{:0.3f}'.format(l)

            if p.is_integer():
                p = int(p)
                if 'label_short_name' in config:
                    p_text = config['label_short_name'][p]
                else:
                    p_text = f'{p}'
            else:
                p_text = '{:0.3f}'.format(p)

            if 'prob_postprocess_func' in config:
                prob = probs[im_idx, 1].item()
                func = config['prob_postprocess_func']
                pp = func(prob)
                pp = pp if p == 1 else 1 - pp
                att_map = att_map if p == 1 else 1. - att_map
                pp = '{:0.3f}'.format(pp)
                disp_str = 'G{}P{}-{}'.format(l_text, p_text, pp)
            else:
                disp_str = 'G{}P{}'.format(l_text, p_text)

            # to uint8
            sim_uint8 = (sim * 255).astype(np.uint8)
            att_map_uint8 = (att_map * 255).astype(np.uint8)
            att_map_uint8 = cv2.applyColorMap(att_map_uint8, cv2.COLORMAP_JET)

            augmented = cv2.addWeighted(sim_uint8, 0.5, att_map_uint8, 0.5, 0.0)
            # utils.plot_image(augmented)

            if labels[im_idx] == preds[im_idx]:
                text_color = (0, 255, 63)
            else:
                text_color = (0, 63, 255)
            augmented = cv2.putText(augmented, disp_str, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    text_color, 2, cv2.LINE_AA)

            augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            augmented = np.moveaxis(augmented, (0, 1, 2), (1, 2, 0)) / 255.
            augmented_images.append(torch.tensor(augmented))

    return augmented_images
