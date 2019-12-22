import argparse

from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default="", help="The path for pretrained model.")

parser.add_argument("--valid_path", type=str, default="./valid",
                    help="""Path to save translation for bleu evaulation. Default is ./valid.""")

parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-gpu.")

parser.add_argument("--task", type=str, choices=["baseline", "odc", "odc_old"])

parser.add_argument("--shared_dir", type=str, default="/tmp",
                    help="""Shared directory across nodes. Default is '/tmp'""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)
    auto_mkdir(args.valid_path)

    if args.task == "baseline":
        from src.tasks.nmt import train
    elif args.task == "odc":
        from src.tasks.odc import train
    else:
        err_info = "Unknown task type %s" % args.task
        raise ValueError(err_info)

    train(args)


if __name__ == '__main__':
    run()
