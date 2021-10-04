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
from torchvision.utils import save_image

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

	# Initialize generator and discriminator
	generator = Generator(opt)
	print("---------------------------- Generator ----------------------------")
	print(generator)
	discriminator = Discriminator(opt)
	print("---------------------------- Discriminator ----------------------------")
	print(discriminator)

	adversarial_loss = torch.nn.BCELoss()

	if cuda:
		generator.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()

	# Optimizers
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	print("---------------------------- Start training ----------------------------")
	for epoch in range(opt.n_epochs):
		for i, batch in enumerate(train_dataloader):
			# Model inputs
			real_A = Variable(batch["A"].type(Tensor))
			real_B = Variable(batch["B"].type(Tensor))

			# Adversarial ground truths
			valid = Variable(Tensor(np.ones((real_A.size(0)))), requires_grad=False)
			fake = Variable(Tensor(np.zeros((real_A.size(0)))), requires_grad=False)
			valid = valid.view(valid.size(0), -1)
			fake = fake.view(fake.size(0), -1)
			
			#  Train Generator
			optimizer_G.zero_grad()

			gen_imgs = generator(real_A)

			g_loss = adversarial_loss(discriminator(gen_imgs), valid)

			g_loss.backward()
			optimizer_G.step()

			
			#  Train Discriminator
			optimizer_D.zero_grad()
			# Measure discriminator's ability to classify real from generated samples

			real_B = real_B.view(real_B.size(0), real_B.size(1), real_B.size(3), real_B.size(4))

			real_loss = adversarial_loss(discriminator(real_B), valid)
			fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
			d_loss = (real_loss + fake_loss) / 2
			
			d_loss.backward()
			optimizer_D.step()
			
			
			print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
			% (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
			)

			batches_done = epoch * len(train_dataloader) + i
			
			
			if batches_done % opt.sample_interval == 0:
				save_image(torch.cat((gen_imgs[0][0], gen_imgs[0][1], gen_imgs[0][2], gen_imgs[0][3], gen_imgs[0][4] \
									, gen_imgs[0][5], gen_imgs[0][6], gen_imgs[0][7], gen_imgs[0][8], gen_imgs[0][9]), 1) \
				, "images/%d.png" % batches_done, nrow=5, normalize=True)
		
		if epoch % opt.save_interval == 0:
			torch.save(generator, 'checkpoints/G_epoch_{}.pth'.format(epoch))
			torch.save(discriminator, 'checkpoints/D_epoch_{}.pth'.format(epoch))

	torch.save(generator, 'checkpoints/G_epoch_final.pth')
	torch.save(discriminator, 'checkpoints/D_epoch_final.pth')