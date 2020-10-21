import torch
from torch import nn
from torchsummary import summary
from common.logger import logger_
from common import seed
from gans.model.apgan.apgan_modules import ConvBlockD
from gans.model.apgan.abs_sub_apgan import AbstractSubAPGAN


class Discriminator(AbstractSubAPGAN):
    def __init__(self, config, device):
        """
        config: the root config should be "d"
        """
        super().__init__(config, device)
        self.downsampler = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def _build_rgb_layer(self):
        """
        Supposed to be called when constructing Discriminator and before training.
        Create all of the necessary rgb layers since when step is changed, old and new rgb layer are gradually changed.
        """
        from_rgb_list = list()
        out_channel_list = [512, 512, 256, 128, 64, 32, 16]
        for out_ch in out_channel_list:
            from_rgb_list.append(nn.Conv2d(in_channels=self.config["num_ch"], out_channels=out_ch,
                                           kernel_size=1, stride=1, padding=0))
        return nn.ModuleList(from_rgb_list)

    def _build_block(self):
        """
        Supposed to be called when constructing Discriminator and before training.
        Here, constructing unchanged structure while training.
        """
        att_pos_idx = self.config["attention_pos_idx"]
        use_std_layer = self.config["use_std_layer"]
        blocks = list()
        # step_id = 0 (out_feat_size = 1)
        blocks.append(ConvBlockD(block_in_ch=512, block_out_ch=512, config=self.config,
                                 for_1st_step=True, with_std=use_std_layer))
        # step_id = 1 (out_feat_size = 4)
        blocks.append(ConvBlockD(block_in_ch=512, block_out_ch=512, config=self.config, with_att=att_pos_idx == 1))
        # step_id = 2 (out_feat_size = 8)
        blocks.append(ConvBlockD(block_in_ch=256, block_out_ch=512, config=self.config, with_att=att_pos_idx == 2))
        # step_id = 3 (out_feat_size = 16)
        blocks.append(ConvBlockD(block_in_ch=128, block_out_ch=256, config=self.config, with_att=att_pos_idx == 3))
        # step_id = 4 (out_feat_size = 32)
        blocks.append(ConvBlockD(block_in_ch=64, block_out_ch=128, config=self.config, with_att=att_pos_idx == 4))
        # step_id = 5 (out_feat_size = 64)
        blocks.append(ConvBlockD(block_in_ch=32, block_out_ch=64, config=self.config, with_att=att_pos_idx == 5))
        # step_id = 6 (out_feat_size = 128)
        blocks.append(ConvBlockD(block_in_ch=16, block_out_ch=32, config=self.config, with_att=att_pos_idx == 6))

        return nn.ModuleList(blocks)

    def _initialize_weight(self, m):
        """
        It is used as an argument in net.apply(fn)
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 1)
            self._store_init_layer_nm(m.__class__.__name__)

    @classmethod
    def check_structure(cls, config, max_step_idx, device):
        init_img_size = 4
        d = cls(config, device)
        logger_.info(f"Registered modules in discriminator")
        logger_.info(d)
        for step_idx in range(max_step_idx + 1):
            img_size = init_img_size * 2 ** step_idx
            noise = torch.randn(1, 1, img_size, img_size)
            logger_.info(f"[D] Summary in step {step_idx}")
            logger_.info(f"\n{summary(d, noise, step_idx=step_idx)}")

    def __forward_for_1st_step(self, x):
        x = self.rgb_layers[0](x)
        x = self.blocks[0](x)
        return x

    def __forward_for_other_steps(self, x, step_idx):
        # alpha-route for new structure
        x_new = self.rgb_layers[step_idx](x)
        x_new = self.blocks[step_idx](x_new)
        x_new = x_new * self.alpha

        # (alpha - 1)-route for prev structure
        x_prev = self.downsampler(x)
        x_prev = self.rgb_layers[step_idx - 1](x_prev)

        # common-route for both structure
        x_mix = x_new + x_prev
        for i in reversed(range(step_idx)):  # flow is like 0, 1->0, 2->1->0
            x_mix = self.blocks[i](x_mix)
        return x_mix

    def forward(self, x, step_idx):
        if step_idx == 0:
            return self.__forward_for_1st_step(x)
        else:
            return self.__forward_for_other_steps(x, step_idx)

