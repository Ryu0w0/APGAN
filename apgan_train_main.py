"""
It trains APGAN and produces a trained generator as *.ptn files.
"""


def initialization():
    """
    Initialising arguments, logger, tensorboard recorder and json files.
    """
    import argparse
    from torch.utils.tensorboard import SummaryWriter
    from common import file_operator as f_op
    from common import logger as log_util
    from common import seed

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_config_key", type=str, default="default_ch3", help="json config file name")
    parser.add_argument("-save_key", type=str, default="apgan",
                        help="Used as a file name of dataset and log files")
    parser.add_argument("-num_fixed_noise", type=int, default=16)
    parser.add_argument("-log_level", type=str, default="INFO")
    # PER X ITERATION
    parser.add_argument("-logging_per_iter", type=int, default=1)
    parser.add_argument("-validate_per_iter", type=int, default=5)
    parser.add_argument("-save_img_per_iter", type=int, default=5)
    parser.add_argument("-save_model_per_iter", type=int, default=5)
    # MODEL and LOSS
    parser.add_argument("-loss_f_type", type=str, default="wgan_gp",
                        help="bce, hinge or wgan_gp")
    # TRAINING
    parser.add_argument("-step_from", type=int, default=0, help="steps run for range($step_from, $num_steps + 1)")
    parser.add_argument("-num_steps", type=int, default=6, help="used such that train from 0 to $num_steps")
    parser.add_argument("-num_img_iter", type=int, default=1,
                        help="number of iteration per step training for a specific resolution")
    parser.add_argument("-batch_size_list", type=str, default="2,2,2,2,2,2,2",
                        help="batch_size from 0 to Nth step")  # "256,256,128,64,32,16,8"
    parser.add_argument("-num_workers", type=int, default=1)
    parser.add_argument("-use_gpu", type=int, default=0)
    parser.add_argument("-is_reproducible", type=int, default=1)
    # PATH
    parser.add_argument("-save_root_path", type=str, default="./files/output/gans")
    parser.add_argument("-load_root_path", type=str, default="./files/load/gans")
    parser.add_argument("-load_dataset_root_path", type=str, default="./files/load/prep/dataset/dataset_patch")
    parser.add_argument("-config_root_path", type=str, default="./gans/model/apgan/config/")
    args = parser.parse_args()

    # convert batch_size list into a list
    args.batch_size_list = [int(s) for s in args.batch_size_list.split(",")]

    # create logger
    args.log_level = log_util.get_log_level_from_name(args.log_level)
    log_root_dir = f"{args.save_root_path}/logs"
    logger_ = log_util.create_logger("main", log_root_dir, args.save_key, args.log_level)
    log_util.logger_ = logger_

    # load config
    config = f_op.load_json(args.config_root_path, f"{args.load_config_key}")
    logger_.info("** CONFIG: GENERATOR **")
    for k, v in config["g"].items():
        logger_.info(f"{k}: {v}")
    logger_.info("** CONFIG: DISCRIMINATOR **")
    for k, v in config["d"].items():
        logger_.info(f"{k}: {v}")

    # create TensorBoard writer
    board_root_dir = f"{args.save_root_path}/board/{args.save_key}"
    f_op.create_folder(board_root_dir)
    log_util.writer_ = SummaryWriter(board_root_dir)

    # set seed
    if args.is_reproducible:
        seed.feed_seed = True

    # logging
    logger_.info("*** ARGUMENTS ***")
    for k, v in args.__dict__.items():
        logger_.info(f"{k}: {v}")

    return args, config


def main():
    args, config = initialization()
    from common.logger import logger_
    from gans.model.apgan.apgan_gp import APGANGP
    from gans.model.apgan import discriminator as d
    from gans.model.apgan import generator as g
    from gans.loss.gan_loss import GANLoss

    # Set device
    logger_.info("*** SET DEVICE ***")
    device = "cpu" if args.use_gpu == 0 else "cuda"
    logger_.info(f"Device is {device}")

    # Create generator and its optimizer
    logger_.info("*** CREATE GENERATOR ***")
    netG = g.Generator(config["g"], device).to(device)
    logger_.info(f"Optimizer: {netG.optimizer}")
    logger_.info("*** CHECK GENERATOR STRUCTURE ***")
    # g.Generator.check_structure(config["g"], max_step_idx=6, device=device)

    # Create the Discriminator and its optimizer
    logger_.info("*** CREATE DISCRIMINATOR ***")
    netD = d.Discriminator(config["d"], device).to(device)
    logger_.info(f"Optimizer: {netD.optimizer}")
    logger_.info("*** CHECK DISCRIMINATOR STRUCTURE ***")
    # d.Discriminator.check_structure(config["d"], max_step_idx=6, device=device)

    # Loss function
    logger_.info("*** CREATE LOSS FUNCTION ***")
    criterion = GANLoss(loss_type=args.loss_f_type, device=device)
    logger_.info(criterion)

    # criterion = nn.BCELoss()
    logger_.info("*** CREATE GAN NETWORK ***")
    gan = APGANGP(netG, netD, criterion, args, config, device)

    # convert weights of models into double
    gan.g = gan.g.float()
    gan.d = gan.d.float()

    # Training
    logger_.info("*** TRAINING ***")
    print("Starting Training Loop...")
    gan.train(step_from=args.step_from, num_steps=args.num_steps)

    # Save model
    # logger_.info("*** SAVE WEIGHTS ***")
    gan.save_model()
    logger_.info("FINISH.")
    exit(0)


if __name__ == '__main__':
    main()


