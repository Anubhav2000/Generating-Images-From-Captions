import torch
from torchvision import transforms
import torchvision
from cfg import d
from torch import autograd
import os
from generator import generator
from discriminator import discriminator
from torch.utils.data import DataLoader
from CUBDataset import CUBDataset
from torch.autograd import Variable
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import pdb

class Helper(object):
    def save_model(self, netD, netG, optD, optG, dir_path, epoch):
        path = dir_path
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/Discriminator_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/Generator_{1}.pth'.format(path, epoch))
        torch.save(optG.state_dict(), '{0}/Generator_Optimizer_{1}.pth'.format(path, epoch))
        torch.save(optD.state_dict(), '{0}/Discriminator_Optimizer_{1}.pth'.format(path, epoch))

    def initializeWieights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def load_model(self, path, epoch, cuda):
        if cuda == False:
            netG = generator()
            netD = discriminator()
        else:
            netG = torch.nn.DataParallel(generator().cuda())
            netD = torch.nn.DataParallel(discriminator().cuda())
        netD.load_state_dict(torch.load(path + '/Discriminator_{0}.pth'.format(epoch)))
        netG.load_state_dict(torch.load(path + '/Generator_{0}.pth'.format(epoch)))
        netD.train()
        netG.train()
        optimD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.99), amsgrad=True)
        optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.99), amsgrad=True)
        optimG.load_state_dict(torch.load(path + '/Generator_Optimizer_{0}.pth'.format(epoch)))
        optimD.load_state_dict(torch.load(path + '/Discriminator_Optimizer_{0}.pth'.format(epoch)))
        return netD, netG, optimD, optimG
    
    def draw_save(self, correct_image, incorrect_image):
            idx = np.random.randint(0, 64, 1)
            image = correct_image[idx,:, :].numpy()
            
# class Logger(object):
#     def __init__(self, vis_screen):
#         self.viz = VisdomPlotter(env_name=vis_screen)
#         self.hist_D = []
#         self.hist_G = []
#         self.hist_Dx = []
#         self.hist_DGx = []
#
#     def log_iteration_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss):
#         print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, fake_loss = %f" %
#               (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, fake_loss))
#         self.hist_D.append(d_loss.data.cpu().mean())
#         self.hist_G.append(g_loss.data.cpu().mean())
#
#     def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
#         print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
#             epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
#             fake_score.data.cpu().mean()))
#         self.hist_D.append(d_loss.data.cpu().mean())
#         self.hist_G.append(g_loss.data.cpu().mean())
#         self.hist_Dx.append(real_score.data.cpu().mean())
#         self.hist_DGx.append(fake_score.data.cpu().mean())
#
#     def plot_epoch(self, epoch):
#         self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
#         self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
#         self.hist_D = []
#         self.hist_G = []
#
#     def plot_epoch_w_scores(self, epoch):
#         self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
#         self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
#         self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
#         self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
#         self.hist_D = []
#         self.hist_G = []
#         self.hist_Dx = []
#         self.hist_DGx = []
#
#     def draw(self, right_images, fake_images):
#         self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64] * 128 + 128)
#         self.viz.draw('real images', right_images.data.cpu().numpy()[:64] * 128 + 128)

# class VisdomPlotter(object):
#
#     def __init__(self, env_name='gan'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#
#     def plot(self, var_name, split_name, x, y, xlabel='epoch'):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=var_name,
#                 xlabel=xlabel,
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)
#
#     def draw(self, var_name, images):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.images(images, env=self.env)
#         else:
#             self.viz.images(images, env=self.env, win=self.plots[var_name])

class Trainer(object):
    def __init__(self, data_dir, batch_size, epochs, save_path, learning_rate, split, is_pretrained):
        self.dev = d()
        self.help = Helper()
        # self.log = Logger('Gan')
        if self.dev.cuda == False:
            self.gen = generator()
            self.disc = discriminator()
        else:
            self.gen = torch.nn.DataParallel(generator().cuda())
            self.disc = torch.nn.DataParallel(discriminator().cuda())

        self.dataset = CUBDataset(data_dir, split=split)
        self.dataLoader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        print("Data Loaded Successfully")
        self.optimD = torch.optim.Adam(self.disc.parameters(), lr=learning_rate, betas=(0.5, 0.99), amsgrad=True)
        self.optimG = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(0.5, 0.99), amsgrad=True)
        
        if(is_pretrained == True):
            self.disc, self.gen, self.optimD, self.optimG = self.help.load_model('checkpoints', 90, self.dev.cuda)
        self.model_dir = save_path
        self.epochs = self.dev.epochs
        self.batch_size = epochs
        self.train()

    def train(self):
        print("Training Started")
        bce = torch.nn.BCELoss()
        mse = torch.nn.MSELoss()
        l1 = torch.nn.L1Loss()
        epoch = 91
        while epoch < self.epochs:
            
            dataiterator = iter(self.dataLoader)
            iteri = 0
            for point in self.dataLoader:
                # print(len(point))
                iteri = iteri + 1
                print(str(epoch) + " " + str(iteri))
                # point = next(dataiterator)
                correct_image = point['correct_image']
                incorrect_image = point['incorrect_image']
                correct_embed = point['correct_embed']
                #---------------------------------------------------------------------
                if(correct_image.size(0)!=self.batch_size):
                    continue
                #For discriminator
                if(self.dev.cuda ==True):
                    correct_image = Variable(correct_image.float()).cuda()
                    incorrect_image = Variable(incorrect_image.float()).cuda()
                    correct_embed = Variable(correct_embed.float()).cuda()
    
                    incorrect_labels = Variable(torch.zeros(self.batch_size)).cuda()
                    # One Sided Label Smoothing
                    correct_labels = torch.FloatTensor(torch.ones(self.batch_size) + -1)
                    correct_labels  = Variable(correct_labels).cuda()
    
                    self.disc.zero_grad()
                    # Right images and right caption
                    output, activations = self.disc(correct_image, correct_embed)
                    correct_loss = bce(output, correct_labels)
                    # Wrong image and right caption
                    output, activations = self.disc(incorrect_image, correct_embed)
                    incorrect_loss = bce(output, incorrect_labels)
    
                    #Generated image and right captions
                    noise = Variable(torch.randn(self.batch_size, 100)).cuda()
                    noise = noise.view(self.batch_size, 100, 1, 1)
                    # Feeding it to the discriminator
                    generated_images = Variable(self.gen(correct_embed, noise)).cuda()
                    output, activations = self.disc(generated_images, correct_embed)
                    generated_loss = torch.mean(output)
                    # Calculating the net loss
                    net_loss = generated_loss + correct_loss + incorrect_loss
                    net_loss.backward()
                    # Taking one more step towards convergence
                    self.optimD.step()
                    # ----------------------------------------------------------------------------
                    #For generator
                    self.gen.zero_grad()
                    noise = Variable(torch.randn(self.batch_size, 100)).cuda()
                    noise = noise.view(self.batch_size, 100, 1, 1)
    
                    generated_images = Variable(self.gen(correct_embed, noise)).cuda()
                    output, generated = self.disc(generated_images, correct_embed)
                    output, real = self.disc(correct_image, correct_embed)
    
                    generated = torch.mean(generated, 0)
                    real = torch.mean(real, 0)
    
                    net_loss = bce(output, correct_labels) + mse(generated, real)*100 + 50*l1(generated_images, correct_image)
                    net_loss.backward()
                    self.optimG.step()
                else:
                    correct_image = Variable(correct_image.float())
                    incorrect_image = Variable(incorrect_image.float())
                    correct_embed = Variable(correct_embed.float())
    
                    incorrect_labels =torch.zeros(correct_image.size(0))
                    # One Sided Label Smoothing
                    correct_labels = torch.ones(correct_image.size(0))
                    correct_labels  = Variable(correct_labels + -1)
    
                    self.disc.zero_grad()
                    # Right images and right caption
                    output, activations = self.disc(correct_image, correct_embed)
                    correct_loss = bce(output, correct_labels)
                    # Wrong image and right caption
                    output, activations = self.disc(incorrect_image, correct_embed)
                    incorrect_loss = bce(output, incorrect_labels)
    
                    #Generated image and right captions
                    noise = Variable(torch.randn(correct_image.size(0), 100))
                    noise = noise.view(self.batch_size, 100, 1, 1)
                    # Feeding it to the discriminator
                    generated_images = Variable(self.gen(correct_embed, noise))
                    output, activations = self.disc(generated_images, correct_embed)
                    generated_loss = torch.mean(output)
                    # Calculating the net loss
                    net_loss = generated_loss + correct_loss + incorrect_loss
                    net_loss.backward()
                    # Taking one more step towards convergence
                    self.optimD.step()
                    # ----------------------------------------------------------------------------
                    #For generator
                    self.gen.zero_grad()
                    noise = Variable(torch.randn(self.batch_size, 100))
                    noise = noise.view(self.batch_size, 100, 1, 1)
    
                    generated_images = Variable(self.gen(correct_embed, noise))
                    output, generated = self.disc(generated_images, correct_embed)
                    output, real = self.disc(correct_image, correct_embed)
    
                    generated = torch.mean(generated, 0)
                    real = torch.mean(real, 0)
    
                    net_loss = bce(output, correct_labels) + mse(generated, real)*100 + 50*l1(generated_images, correct_image)
                    net_loss.backward()
                    self.optimG.step()
                    
            if epoch%10==0:
                self.help.save_model(self.disc, self.gen, self.optimD, self.optimG, 'checkpoints', epoch)
            if epoch%2==0 :
                path = str(epoch) + ' images'
                # self.log.draw(correct_image, incorrect_image)
            epoch+=1
