from torch.nn import functional as F
from gans.model.apgan.abs_apgan import AbstractAPGAN
from common.logger import logger_


class APGAN(AbstractAPGAN):
    def __init__(self, generator, discriminator, loss_f, dataset, args, config, device, evaluator):
        super().__init__(generator, discriminator, loss_f, dataset, args, config, device, evaluator)
        self.gp_weight = 10
        # training
        self._train_generator_step = self._train_g_step
        self._train_discriminator_step = self._train_d_step

    def train(self, num_steps):
        for i in range(num_steps):
            logger_.info(f"*** START TRAIN STEP {i} ***")
            self.set_loader(step_idx=i)
            self.__train_step(cur_step=i)

    def __train_step(self, cur_step):
        num_trained_images = 0
        num_iter = self.get_num_iter(cur_step)
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

            del real_img
            del fake_img
            self.logging_if_necessary(cur_step, i+1, num_iter, loss_d, loss_g)
            self.validate_if_necessary(cur_step, i)
            # self.save_model_if_necessary()
            self.save_img_if_necessary(cur_step, i, real_img)
            self.cur_global_step += 1

    def _train_d_step(self, real_img, fake_img, _, step_idx):
        ## Train with all-real batch
        self.d.zero_grad()
        # Forward pass real batch through D
        output_real = self.d(real_img, step_idx).view(-1)
        # Calculate loss on all-real batch
        loss_d_real, d_x = self.loss_f(output_real, target_is_real=True, for_discriminator=True)
        # Calculate gradients for D in backward pass
        loss_d_real.backward()
        # d_x = output_real.mean().item()

        ## Train with all-fake batch
        # Classify all fake batch with D
        output_fake = self.d(fake_img.detach(), step_idx=step_idx).view(-1)
        # Calculate D's loss on the all-fake batch
        loss_d_fake, _ = self.loss_f(output_fake, target_is_real=False, for_discriminator=True)
        # Calculate the gradients for this batch
        loss_d_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        loss_d = loss_d_real + loss_d_fake
        # Update D
        self.d.optimizer.step()

        return loss_d

    def _train_g_step(self, fake_img, step_idx):
        self.g.zero_grad()
        # feedforward again to get updated D
        output_fake = self.d(fake_img, step_idx).view(-1)
        # Calculate G's loss based on this output
        loss_g, _ = self.loss_f(output_fake, target_is_real=True, for_discriminator=False)
        # Calculate gradients for G
        loss_g.backward()
        # Update G
        self.g.optimizer.step()

        return loss_g
