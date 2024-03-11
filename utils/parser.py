import argparse

def set_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.00003, type=float)
    parser.add_argument('--image_size', default=256, type=int)
    
    return parser.parse_args()