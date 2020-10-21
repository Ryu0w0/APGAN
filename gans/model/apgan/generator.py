import torch
from torch import nn
from torchsummary import summary
from common import seed
from common.logger import logger_
from gans.model.apgan.abs_sub_apgan import AbstractSubAPGAN
from gans.model.apgan.apgan_modules import ConvBlockG


class Generator(AbstractSubAPGAN):
    """
    Using nearest neighbor filtering
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.latent_space_size = config["latent_space_size"]
        self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")

    def _build_rgb_layer(self):
        to_rgb_list = list()
        in_channel_list = [512, 512, 256, 128, 64, 32, 16]
        for in_ch in in_channel_list:
            to_rgb_list.append(nn.Conv2d(in_channels=in_ch, out_channels=self.config["num_ch"],
                                         kernel_size=1, stride=1, padding=0))
        return nn.ModuleList(to_rgb_list)

    def _build_block(self):
        att_pos_idx = self.config["attention_pos_idx"]
        blocks = list()
        # step_id = 0 (out_feat_size = 4)
        blocks.append(ConvBlockG(block_in_ch=100, block_out_ch=512, config=self.config, for_1st_step=True))
        # step_id = 1 (out_feat_size = 8)
        blocks.append(ConvBlockG(block_in_ch=512, block_out_ch=512, config=self.config, with_att=att_pos_idx == 1))
        # step_id = 2 (out_feat_size = 16)
        blocks.append(ConvBlockG(block_in_ch=512, block_out_ch=256, config=self.config, with_att=att_pos_idx == 2))
        # step_id = 3 (out_feat_size = 32)
        blocks.append(ConvBlockG(block_in_ch=256, block_out_ch=128, config=self.config, with_att=att_pos_idx == 3))
        # step_id = 4 (out_feat_size = 64)
        blocks.append(ConvBlockG(block_in_ch=128, block_out_ch=64, config=self.config, with_att=att_pos_idx == 4))
        # step_id = 5 (out_feat_size = 128)
        blocks.append(ConvBlockG(block_in_ch=64, block_out_ch=32, config=self.config, with_att=att_pos_idx == 5))
        # step_id = 6 (out_feat_size = 256)
        blocks.append(ConvBlockG(block_in_ch=32, block_out_ch=16, config=self.config, with_att=att_pos_idx == 6))

        return nn.ModuleList(blocks)

    def sample_noise(self, num_noise=1):
        # sample from standard normal distribution
        seed.seed_everything()
        noise = torch.randn(num_noise, self.latent_space_size, 1, 1, device=self.device)
        return noise

    @classmethod
    def check_structure(cls, config, max_step_idx, device):
        g = cls(config, device)
        logger_.info(f"Registered modules in generator.")
        logger_.info(g)
        for step_idx in range(max_step_idx + 1):
            logger_.info(f"[G] Summary in step {step_idx}")
            logger_.info(f"\n{summary(g, g.sample_noise(), step_idx=step_idx)}")

    def _initialize_weight(self, m):
        """
        It is used as an argument in net.apply(fn)
        """
        seed.seed_everything()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 1)
            self._store_init_layer_nm(m.__class__.__name__)

    def __forward_for_1st_step(self, x):
        """
        First step does not have branch-route for the residual training.
        """
        x = self.blocks[0](x)
        x = self.rgb_layers[0](x)
        return x

    def __forward_for_other_steps(self, x, step_idx):
        """
        In later than the first step, the residual training is introduced.
        """
        # common-route for both structure
        for i in range(step_idx):
            x = self.blocks[i](x)

        # alpha-route for new structure
        x_new = self.blocks[step_idx](x)
        x_new = self.rgb_layers[step_idx](x_new)
        x_new = x_new * self.alpha

        # (1-alpha)-route for previous structure
        x_prev = self.upsampler(x)
        x_prev = self.rgb_layers[step_idx - 1](x_prev)
        x_prev = x_prev * (1 - self.alpha)
        out = x_new + x_prev
        return out

    def forward(self, x, step_idx):
        if step_idx == 0:
            return self.__forward_for_1st_step(x)
        else:
            return self.__forward_for_other_steps(x, step_idx)
