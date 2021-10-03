import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import *
from torch.utils.data import DataLoader
from dataset import ImageDataset
import time
import numpy as np
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False

def sample_images(batches_done, opt, data):
	"""Saves a generated sample from the validation set"""
	imgs = next(iter(data))
	real_A = Variable(imgs["B"].type(Tensor))
	real_B = Variable(imgs["A"].type(Tensor))
	fake_B = generator(real_A)
	img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
	save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

def train(opt):
	# Loss function
	adversarial_loss = torch.nn.BCELoss()

	# Initialize generator and discriminator
	generator = Generator(opt)
	discriminator = Discriminator(opt)

	if cuda:
		generator.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()

	

	# Optimizers
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	transforms_ = [
		transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize((0.5), (0.5)),
	]

	dataset = ImageDataset(opt.dataset, transforms_=transforms_)

	total_size = len(dataset)
	training_set_size = int(total_size*0.9)
	validation_set_size = total_size - training_set_size

	training_set, validation_set = torch.utils.data.random_split(dataset, [training_set_size, validation_set_size])

	train_dataloader = DataLoader(
		training_set,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.n_cpu,
	)

	valid_dataloader = DataLoader(
		validation_set,
		batch_size=10,
		shuffle=True,
		num_workers=1,
	)

	for epoch in range(opt.n_epochs):
		for i, batch in enumerate(train_dataloader):
			print("epoch:",epoch)