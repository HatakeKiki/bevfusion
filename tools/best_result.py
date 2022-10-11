import json
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file', type=str, default=None, help='json file')
    parser.add_argument('--metric', type=str, default='map', help='metric for choosing best epoch')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_config()
    path = args.file
    with open(path, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        process = json.loads(line)
        if 'mode' not in process.keys():
            continue
        if process['mode'] == 'val':
            results.append(process)

    best_epoch_metric = 0
    best_result = None
    objects = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    metrics = ['ap_dist_4.0', 'ap_dist_2.0', 'ap_dist_1.0', 'ap_dist_0.5']
    prefix = 'object/'
    # metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
    for result in results:
        if args.metric == 'all':
            new_metric = result['object/map'] + result['object/nds']
        else:
            new_metric = result['object/%s' % args.metric]
        if new_metric > best_epoch_metric:
            best_epoch_metric = new_metric
            best_result = result

    aps = {}
    for object in objects:
        ap = 0
        for metric in metrics:
            ap += best_result['%s%s_%s' % (prefix, object, metric)]
        ap /= len(metrics)
        aps.update({object : ap})
    map = best_result['object/map']
    nds = best_result['object/nds']
    print('============= Best Result in Epoch: %d =============' % best_result['epoch'])
    print('mAP: %.4f' % map)
    print('NDS: %.4f' % nds)

    print('%.4f' % map)
    print('%.4f' % nds)
    # print('----------------------------------------------------')
    for key in aps.keys():
        print('%.4f' % aps[key])
        # print('%s: %.4f' % (key, aps[key]))
