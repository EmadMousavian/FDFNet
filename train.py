import torch
import argparse
from tools.config import cfg, cfg_from_yaml_file
from model.dataset import build_dataloader
from pathlib import Path
from torch.optim import SGD
import torch.nn as nn
from model.FDFNet import build_network, set_random_seed
from tools.train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path to start from')
    parser.add_argument('--last_epoch', type=int, default=None, help='last epoch that ckpt create for continuing training')
    parser.add_argument('--max_training_time', type=int, default=None, help='maximum training time (second)')
    args = parser.parse_args()
    cfg.TAG = Path(args.cfg_file).stem  # remove 'cfgs' and 'xxxx.yaml'
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE

    output_dir = cfg.ROOT_DIR / 'output' / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print('**********************Start logging**********************')
    for key, val in vars(args).items():
        print('{:16} {}'.format(key, val))
    print("----------- Create dataloader & network & optimizer -----------")
    for mod in ["mono", "poly"]:
        loader = build_dataloader(
            data_path=cfg.DATA_CONFIG[mod],
            batch_size=args.batch_size,
            workers=args.workers,
            mode="train",
            types=mod
        )
        val_loader = build_dataloader(
            data_path=cfg.DATA_CONFIG[mod],
            batch_size=args.batch_size,
            workers=args.workers,
            mode="val",
            types=mod
        )

        set_random_seed(42)
        network = build_network(model_cfg=cfg.MODEL)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("let use", torch.cuda.device_count(), "gpu")
            network = nn.DataParallel(network)
        network = network.to(device)

        # build optimizer and scheduler
        optimizer = SGD(network.parameters(), lr=cfg.OPTIMIZATION.LR, momentum=cfg.OPTIMIZATION.MOMENTUM,
                        weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)
        lambda1 = lambda epoch: ((1 - (epoch / args.epochs)) ** 0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda1)

        start_epoch = 1
        if args.ckpt is not None:
            if torch.cuda.device_count() == 1:
                network = nn.DataParallel(network, device_ids=[0])
            network.load_state_dict(torch.load(args.ckpt))
            if args.last_epoch is not None:
                start_epoch = args.last_epoch + 1
            else:
                raise ValueError
            print('********************** checkpoint loaded **********************')
            print(f'********** continue training from epoch {start_epoch} *********')
            for epoch in range(1, start_epoch):
                scheduler.step()

        print('**********************Start training %s/%s**********************' % (cfg.TAG, args.extra_tag))
        class_weights = cfg.OPTIMIZATION.class_weights[mod]
        class_weights = torch.Tensor(class_weights).to(device)
        # Loss functions
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_model(
            mode=mod,
            dataloader=loader,
            val_dataloader=val_loader,
            network=network,
            start_epoch=start_epoch,
            num_epoch=args.epochs,
            device=device,
            scheduler=scheduler,
            criterion=criterion,
            optimizer=optimizer,
            output_path=ckpt_dir,
            max_time=args.max_training_time
        )


if __name__ == '__main__':
    main()
