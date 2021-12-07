import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch TCR')
    #parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
    parser.add_argument('--with_tcr', type=int, default=0, help='training batch size')
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    #parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate. Default=0.01')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/model_checkpoint.pt', help='Model checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loader to use')
    #parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--weight_tcr', type=float, default=0.01, help='random seed to use. Default=123')
    return parser
