from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom


def data_provider(args, flag):
    shuffle = flag == "train"
    stride = 1 if flag == "train" else args.stride
    dataset = Dataset_Custom(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        scale=args.scale,
        timeenc=1 if args.embed == "timeF" else 0,
        freq=args.freq,
        stride=stride,
        series_id=args.series_id,
        top_k_events=args.top_k_events,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader
