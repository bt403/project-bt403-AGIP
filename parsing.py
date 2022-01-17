import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch TCR for image denoising')
    parser.add_argument('--with_tcr', type=int, default=0, help='determine if tcr will be used. 1 if True.')
    parser.add_argument('--val_un', type=int, default=0, help='determine if validation will be done in the validation Coco dataset. 1 if True.')
    parser.add_argument('--val_kodak', type=int, default=0, help='determine if validation will be done in the validation Kodak24. 1 if True.')    
    parser.add_argument('--val_mcmaster', type=int, default=0, help='determine if validation will be done in the validation McMaster 1 if True.')    
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size for supervised')
    parser.add_argument('--batch_size_un', type=int, default=10, help='training batch size for unsupervised')
    parser.add_argument('--noise_level', type=int, default=25, help='noise level for validation run')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/model_checkpoint.pt', help='Model checkpoint')
    parser.add_argument('--resume', type=int, default=0, help='determine if the training will resume from an existing --checkpoint. 1 if True.')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for data loader to use')
    parser.add_argument('--weight_tcr', type=float, default=10, help='weight value for the TCR loss')
    return parser
