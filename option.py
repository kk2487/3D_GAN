import argparse

def parse_opts():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default="test", help='# train, test')
	parser.add_argument('--image_size', type=int, default=224, help='size of generating images')
	parser.add_argument('--input_c', type=int, default=3, help='# image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--epoch', type=int, default=100, help='# number of epochs')
	parser.add_argument('--batch_size', type=int, default=8, help='# input batch size')
	args = parser.parse_args()
	return args
