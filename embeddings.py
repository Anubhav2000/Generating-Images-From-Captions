import torch
from torch.autograd import Variable

from loop import Helper
from CUBDataset import CUBDataset
import matplotlib.pyplot
from torch.utils.data import DataLoader
from cfg import d

if __name__=='__main__':
	help = Helper()
	netD, netG, optimD, optimG = help.load_model('checkpoints', 190, True)
	dataset = CUBDataset("birds.hdf5", split='test')
	dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, pin_memory=True)
	dev = d()
	for sample in dataLoader:
		correct_embed = sample['correct_embed']
		correct_image = sample['correct_image']
		caption = sample['caption']
		
		if(dev.cuda==True):
		 	correct_image = Variable(correct_image.float()).cuda()
			correct_embed = Variable(correct_embed.float()).cuda()
			noise = Variable(torch.randn(1, 100)).cuda()
			noise = noise.view(1, 100, 1, 1)
			fake_image = Variable(netG(noise, correct_embed)).cuda()
			fake_image = fake_image.data.cpu().numpy()
			incorrect_image = incorrect_image.data.cpu().numpy()
			plotPoint(fake_image, incorrect_image, caption)
