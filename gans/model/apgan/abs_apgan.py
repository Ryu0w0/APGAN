from abc import ABCMeta
import numpy as np
import torch
from torchvision import datasets as dset
from torchvision import transforms as transforms
from torchvision import utils as vutils
from common import file_operator as f_op
from common.logger import logger_, writer_
from common import seed


class AbstractAPGAN(metaclass=ABCMeta):
    def __init__(self, generator, discriminator, loss_f, args, config, device):
        self.g = generator
        self.d = discriminator
        self.loss_f = loss_f
        self.__loader = None  # set when starting training
        self.loader_iter = None  # provide data access only through loader_iter
        self.args = args
        self.config = config
        self.device = device
        # training
        self._train_generator_step = None  # function
        self._train_discriminator_step = None  # function
        # stats, visualization
        self.cur_global_step = 1  # increment every iteration
        self.fixed_noise = self.get_fixed_noise(args.num_fixed_noise)

    def set_loader(self, step_idx):
        seed.seed_everything(local_seed=step_idx)
        image_size = int(4 * 2 ** step_idx)
        dataset = dset.ImageFolder(root="./files/input/dataset",
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       # pixel values are converted from [0, 1] to [-1, 1]
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size_list[step_idx],
                                             shuffle=True)
        self.__loader = loader
        self.renew_loader_iter()
        logger_.info(f"Set loader with batch size: {self.args.batch_size_list[step_idx]}")
        logger_.info(f"Number of batches: {len(self.__loader)}")

    def renew_loader_iter(self):
        self.loader_iter = iter(self.__loader)

    def get_next_real_img(self):
        data = next(self.loader_iter, None)
        # In case dataset in a loader is used up
        if data is None:
            logger_.info(f"Renew dataset.")
            self.renew_loader_iter()
            data = next(self.loader_iter)
        real_img = data[0].to(self.device)
        return real_img

    def get_fixed_noise(self, num_fixed_noise):
        # generate fixed noise to use visualization
        seed.seed_everything()
        fixed_noise = self.g.sample_noise(num_noise=num_fixed_noise)
        return fixed_noise

    def get_num_iter(self, step_id):
        """
        First step train for a half of other steps.
        """
        num_img_iter = self.args.num_img_iter / 2 if step_id == 0 else self.args.num_img_iter
        num_iter = np.ceil(num_img_iter / self.args.batch_size_list[step_id])
        return int(num_iter)

    def update_alpha(self, num_trained_images):
        """
        Increase alpha according to the number of trained images such that it reaches 1
          when a half of args.num_img_iter are used for training.
        """
        alpha = min(num_trained_images/(self.args.num_img_iter/2), 1)
        writer_.add_scalar(tag="alpha", scalar_value=alpha, global_step=self.cur_global_step)
        self.g.set_alpha(alpha)
        self.d.set_alpha(alpha)

    def logging_if_necessary(self, cur_step, cur_local_step, num_iter, loss_d, loss_g):
        # Output training stats
        if self.cur_global_step % self.args.logging_per_iter == 0:
            logger_.info('[%d][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                         % (self.cur_global_step, cur_step, self.args.num_steps, cur_local_step,
                            num_iter, loss_d.item(), loss_g.item()))
            writer_.add_scalar(tag="err/d", scalar_value=loss_d.item(), global_step=self.cur_global_step)
            writer_.add_scalar(tag="err/g", scalar_value=loss_g.item(), global_step=self.cur_global_step)

    def save_img_if_necessary(self, step_id, cur_local_step, real_img):
        if self.cur_global_step % self.args.save_img_per_iter == 0:
            self.save_synthetic_image(step_id)
            logger_.info(f"{[self.cur_global_step]} Save a grid of synthetic images")
        if cur_local_step == 0:
            self.save_real_image(real_img, step_id)
            logger_.info(f"{[self.cur_global_step]} Save a grid of real images")

    def save_model_if_necessary(self, step_id, cur_local_step):
        if self.args.save_model_per_iter != -1:
            if self.cur_global_step % self.args.save_model_per_iter == 0:
                suffix = f"_{step_id}_{self.cur_global_step}_{cur_local_step}"
                self.save_model(suffix)
                logger_.info(f"{[self.cur_global_step]} Save model")

    def get_size_z(self):
        """
        Obtain the latent space size from weight size of generator.
        """
        return self.g.get_size_z()

    def save_model(self, suffix=""):
        target_root_path = f"{self.args.save_root_path}/model/{self.args.save_key}"
        f_op.create_folder(target_root_path)
        torch.save(self.g.state_dict(), f"{target_root_path}/g{suffix}.ptn")
        # torch.save(self.d.state_dict(), f"{target_root_path}/d{suffix}.ptn")
        logger_.info(f"saved in {target_root_path}")

    def save_synthetic_image(self, step_id):
        with torch.no_grad():
            fake = self.g(self.fixed_noise, step_id).detach().cpu()
            fake = (fake * 0.5) + 0.5  # roll back normalization
            img_array = vutils.make_grid(fake, padding=2, normalize=False).cpu().numpy()
            img_array = np.transpose(img_array, (1, 2, 0))
            save_image_path = self.args.save_root_path + f"/images"
            f_op.save_as_jpg(img_array * 255, save_root_path=save_image_path,
                             save_key=self.args.save_key,
                             file_name=f"fake_step{step_id}_w{fake.size(2)}_{self.cur_global_step}")

    def save_real_image(self, real_images, step_id):
        b, c, w, h = real_images.shape
        real_images = self.change_rgb2bgr(real_images) if c == 3 else real_images
        real_images = (real_images * 0.5) + 0.5  # roll back normalization
        save_image_path = self.args.save_root_path + f"/images"
        img_array = vutils.make_grid(real_images, padding=2, normalize=False).cpu().numpy()
        img_array = np.transpose(img_array, (1, 2, 0))
        f_op.save_as_jpg(img_array * 255, save_root_path=save_image_path,
                         save_key=self.args.save_key, file_name=f"real_step{step_id}_{w}")

    @staticmethod
    def change_rgb2bgr(tensor_img):
        return tensor_img[:, [2, 1, 0], :, :]
