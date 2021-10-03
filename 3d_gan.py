import os
import argparse
from train import *
from test import *

def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default="train", help="train, test")
	parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
	parser.add_argument("--save_interval", type=int, default=10, help="number of epochs of training")
	parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
	parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
	parser.add_argument("--b1", type=float, default=0.1, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.1, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
	parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
	parser.add_argument("--channels", type=int, default=1, help="number of image channels")
	parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
	parser.add_argument("--dataset", type=str, default="C:/Users/hongze/3d_resnet/train_data/night_traindata/", help="data path")
	args = parser.parse_args()
	return args


opt = parse_opts()
# print(opt)

if __name__ == '__main__':

	cuda = True if torch.cuda.is_available() else False
	
	if(opt.mode == 'train'):
		os.makedirs("images", exist_ok=True)
		os.makedirs("checkpoints", exist_ok=True)
		train(opt)


	#if(opt.mode == 'test'):

	