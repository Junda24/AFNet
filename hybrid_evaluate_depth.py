from __future__ import absolute_import, division, print_function
from open3d import *
import os
import cv2
import numpy as np
import torch
from utils import write_ply, backproject_depth, v, npy, Thres_metrics_np

cv2.setNumThreads(
    0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def compute_errors_perimage(gt, pred, min_depth, max_depth):
    valid_mask = (gt > min_depth) & (gt < max_depth)
    epe = np.mean(np.abs(gt[valid_mask] - pred[valid_mask]))
    abs_rel = np.mean(np.abs(gt[valid_mask] - pred[valid_mask]) / gt[valid_mask])
    sq_rel = np.mean(((gt[valid_mask] - pred[valid_mask])**2) / gt[valid_mask])

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

    return {
        'abs_rel':abs_rel.item(),
        'sq_rel':sq_rel.item(),
        'rmse':rmse.item(),
        'rmse_log':rmse_log.item(),
        'a1':a1.item(),
        'a2':a2.item(),
        'a3':a3.item(),
        'log10':log10.item(),
        'valid_number':1.0,
        'abs_diff':epe.item()        
    }

def compute_errors(gt, pred, disable_median_scaling, min_depth, max_depth,
                   interval):
    """Computation of error metrics between predicted and ground truth depths
    """
    # if not disable_median_scaling:
    #     ratio = np.median(gt) / np.median(pred)
    #     pred *= ratio

    # pred[pred < min_depth] = min_depth
    # pred[pred > max_depth] = max_depth
    mask = np.logical_and(gt > min_depth, gt < max_depth)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt[mask] - pred[mask]) / gt[mask])
    print('1',abs_rel)
    abs_rel_2 = np.sum(np.abs(gt[mask] - pred[mask]) / gt[mask])/np.sum(mask.astype(np.float32))

    print('2', abs_rel_2)
    abs_diff = np.mean(np.abs(gt - pred))
    # abs_diff_median = np.median(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    # mask = np.ones_like(pred)
    # thre1 = Thres_metrics_np(pred, gt, mask, 1.0, 0.2)
    # thre3 = Thres_metrics_np(pred, gt, mask, 1.0, 0.5)
    # thre5 = Thres_metrics_np(pred, gt, mask, 1.0, 1.0)

    result = {}
    result['abs_rel'] = abs_rel
    result['sq_rel'] = sq_rel
    result['rmse'] = rmse
    result['rmse_log'] = rmse_log
    result['log10'] = log10
    result['a1'] = a1
    result['a2'] = a2
    result['a3'] = a3
    result['abs_diff'] = abs_diff
    result['total_count'] = 1.0

    return result

def compute_errors1(gt, pred, disable_median_scaling, min_depth, max_depth,
                   interval):
    """Computation of error metrics between predicted and ground truth depths
    """
    # if not disable_median_scaling:
    #     ratio = np.median(gt) / np.median(pred)
    #     pred *= ratio

    # pred[pred < min_depth] = min_depth
    # pred[pred > max_depth] = max_depth

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))
    # abs_diff_median = np.median(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    # mask = np.ones_like(pred)
    # thre1 = Thres_metrics_np(pred, gt, mask, 1.0, 0.2)
    # thre3 = Thres_metrics_np(pred, gt, mask, 1.0, 0.5)
    # thre5 = Thres_metrics_np(pred, gt, mask, 1.0, 1.0)

    result = {}
    result['abs_rel'] = abs_rel
    result['sq_rel'] = sq_rel
    result['rmse'] = rmse
    result['rmse_log'] = rmse_log
    result['log10'] = log10
    result['a1'] = a1
    result['a2'] = a2
    result['a3'] = a3
    result['abs_diff'] = abs_diff
    result['total_count'] = 1.0

    return result


    # return abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3, abs_diff, abs_diff_median, thre1, thre3, thre5


def evaluate_depth_maps(results, config, do_print=False):
    errors = []

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    print('eval against gt depth map of size: %sx%d' %
          (results[0][1].shape[0], results[0][1].shape[1]))
    for i in range(len(results)):
        if i % 100 == 0:
            print('evaluation : %d/%d' % (i, len(results)))

        gt_depth = results[i][1]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = results[i][0]
        filename = results[i][2]
        inv_K = results[i][3]
        if gt_width != pred_depth.shape[1] or gt_height != pred_depth.shape[0]:
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height),
                                    interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(gt_depth > config.MIN_DEPTH,
                              gt_depth < config.MAX_DEPTH)
        if not mask.sum():
            continue

        ind = np.where(mask.flatten())[0]
        if config.vis:
            cam_points = backproject_depth(pred_depth, inv_K, mask=False)
            cam_points_gt = backproject_depth(gt_depth, inv_K, mask=False)
            write_ply('%s/%s_pred.ply' % (config.save_dir, filename),
                      cam_points[ind])
            write_ply('%s/%s_gt.ply' % (config.save_dir, filename),
                      cam_points_gt[ind])

        dataset = filename.split('_')[0]
        interval = (935 - 425) / (128 - 1)  # Interval value used by MVSNet
        errors.append(
            (compute_errors(gt_depth[mask], pred_depth[mask],
                            config.disable_median_scaling, config.MIN_DEPTH,
                            config.MAX_DEPTH, interval), dataset, filename))

    with open('%s/errors.txt' % (config.save_dir), 'w') as f:
        for x, _, fID in errors:
            tex = fID + ' ' + ' '.join(['%.3f' % y for y in x])
            f.write(tex + '\n')

    np.save('%s/error.npy' % config.save_dir, errors)
    results = {}
    all_errors = [x[0] for x in errors]

    print(f"total example evaluated: {len(all_errors)}")
    all_mean_errors = np.array(all_errors).mean(0)
    if do_print:
        print("\n all")
        print("\n  " +
              ("{:>8} | " *
               13).format("abs_rel", "sq_rel", "log10", "rmse", "rmse_log",
                          "a1", "a2", "a3", "abs_diff", "abs_diff_median"))
        print(("&{: 8.3f}  " * 13).format(*all_mean_errors.tolist()) + "\\\\")

    error_names = [
        "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3",
        "abs_diff", "abs_diff_median", "thre1", "thre3", "thre5"
    ]
    results['depth'] = {'error_names': error_names, 'errors': all_mean_errors}

    errors_per_dataset = {}
    for x in errors:
        key = x[1]
        if key not in errors_per_dataset:
            errors_per_dataset[key] = [x[0]]
        else:
            errors_per_dataset[key].append(x[0])
    if config.print_per_dataset_stats:
        for key in errors_per_dataset.keys():
            errors_ = errors_per_dataset[key]
            mean_errors = np.array(errors_).mean(0)

            print("\n dataset %s: %d" % (key, len(errors_)))
            print("\n  " +
                  ("{:>8} | " *
                   13).format("abs_rel", "sq_rel", "log10", "rmse", "rmse_log",
                              "a1", "a2", "a3", "abs_diff", "abs_diff_median",
                              "thre1", "thre3", "thre5"))
            print(("&{: 8.3f}  " * 13).format(*mean_errors.tolist()) + "\\\\")

    print("\n-> Done!")
    return results
