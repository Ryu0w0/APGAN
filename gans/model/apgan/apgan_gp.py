import logging
import torch
from torch.nn import functional as F
from gans.model.apgan.abs_apgan import AbstractAPGAN
from common.logger import logger_, timer
from torch.autograd import grad as torch_grad


class APGANGP(AbstractAPGAN):
    def __init__(self, generator, discriminator, loss_f, args, config, device):
        super().__init__(generator, discriminator, loss_f, args, config, device)
        self.gp_weight = 10
        # training
        self._train_generator_step = self._train_g_step_wgan_gp
        self._train_discriminator_step = self._train_d_step_wgan_gp

    def train(self, step_from, num_steps):
        for i in range(step_from, num_steps + 1):
            logger_.info(f"*** START TRAIN STEP {i} ***")
            self.set_loader(step_idx=i)
            self.__train_step(cur_step=i)

    def __train_step(self, cur_step):
        num_trained_images = 0
        num_iter = self.get_num_iter(cur_step)
        with timer("iteration", logger_=logger_, level=logging.INFO):
            for i in range(num_iter):
                # get real images
                real_img = self.get_next_real_img()
                # get batch size and update alpha
                batch_size = real_img.size(0)
                num_trained_images += batch_size
                self.update_alpha(num_trained_images)
                # get fake images
                noise = self.g.sample_noise(num_noise=batch_size)
                fake_img = self.g(noise, step_idx=cur_step)
                # resize real images into the same size with fake ones
                real_img = F.adaptive_avg_pool2d(real_img, fake_img.shape[2:4])  # change size of real_image
                # training and get loss
                loss_d = self._train_discriminator_step(real_img, fake_img, batch_size, step_idx=cur_step)
                loss_g = self._train_generator_step(fake_img, step_idx=cur_step)
                self.logging_if_necessary(cur_step, i+1, num_iter, loss_d, loss_g)
                self.save_model_if_necessary(cur_step, i)
                self.save_img_if_necessary(cur_step, i, real_img)
                self.cur_global_step += 1

    def _train_d_step_wgan_gp(self, real_img, fake_img, batch_size, step_idx):
        self.d.zero_grad()
        # get output from real and fake images from D
        output_real = self.d(real_img, step_idx=step_idx).view(-1)
        output_fake = self.d(fake_img.detach(), step_idx=step_idx).view(-1)
        # get gradient penalty
        gradient_penalty = self._gradient_penalty(real_img, fake_img, batch_size, step_idx)
        # calculate d loss
        loss_d = output_fake.mean() - output_real.mean() + gradient_penalty
        # calculate update and apply into weights
        loss_d.backward()
        self.d.optimizer.step()

        return loss_d

    def _train_g_step_wgan_gp(self, fake_img, step_idx):
        self.g.zero_grad()
        # get output from fake images
        output_fake = self.d(fake_img, step_idx=step_idx).view(-1)
        # calculate loss for generator
        # change minimum problem into maximum problem since generator wanna maximize Wassertin distance
        loss_g = - output_fake.mean()
        # Calculate gradients for G and apply into weights
        loss_g.backward()
        self.g.optimizer.step()

        return loss_g

    def _gradient_penalty(self, real_img, fake_img, batch_size, step_idx):
        """
        Basis of this function is retrieved from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
        """
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_img)
        interpolated = alpha * real_img.data + (1 - alpha) * fake_img.data
        interpolated = interpolated.requires_grad_().to(device=self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.d(interpolated, step_idx=step_idx)

        # Calculate gradients of probabilities with respect to examples
        # [0]: get the output of partial derivative w.r.t the first variable (only one variable used though)
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device=self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
