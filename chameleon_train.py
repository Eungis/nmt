import os
import argparse
import random
import pprint
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as custom_optim
from torch.utils.data import DataLoader
from chameleon.base_dataset import Vocabulary, TranslationDataset, TranslationCollator
from chameleon.models.seq2seq import Seq2Seq
from chameleon.models.transformer import Transformer
from chameleon.ignite_engine import MLEEngine, MRTEngine
from chameleon.trainer import Trainer
from loguru import logger as log
from chameleon.log import set_logger

set_logger(source="Chameleon", diagnose=True)


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument("--load_fn", required=True, help="Model file name to continue.")
    p.add_argument(
        "--model_fn",
        required=not is_continue,
        help="Model file name to save. Additional information would be annotated to the file name.",
    )
    p.add_argument("--train_fn", required=not is_continue, help="Training Dataset filename.")
    p.add_argument("--valid_fn", required=not is_continue, help="Validation Dataset filename.")
    p.add_argument("--with_text", default=0, help="Return raw text inside the output of the batch.")
    p.add_argument(
        "--src_tgt",
        default="koen",
        type=str,
        help="Source and target language pair. koen means korean to english. Default=%(default)s.",
    )
    p.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s",
    )
    p.add_argument("--use_mps", action="store_true", help="Use mps backends in M1 chip equipped Mac.")
    p.add_argument(
        "--off_autocast",
        action="store_true",
        help="Turn-off Automatic Mixed Precision (AMP), which speed-up training.",
    )
    p.add_argument(
        "--batch_size", type=int, default=32, help="Mini batch size for gradient descent. Default=%(default)s"
    )
    p.add_argument("--n_epochs", type=int, default=20, help="Number of epochs to train. Default=%(default)s")
    p.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s",
    )
    p.add_argument(
        "--init_epoch",
        required=is_continue,
        type=int,
        default=1,
        help="Set initial epoch number, which can be useful in continue training. Default=%(default)s",
    )
    p.add_argument(
        "--max_length", type=int, default=128, help="Maximum length of the training sequence. Default=%(default)s"
    )
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default=%(default)s")
    p.add_argument(
        "--word_vec_size", type=int, default=512, help="Word embedding vector dimension. Default=%(default)s"
    )
    p.add_argument("--hidden_size", type=int, default=768, help="Hidden size of LSTM. Default=%(default)s")
    p.add_argument("--n_layers", type=int, default=4, help="Number of layers in LSTM. Default=%(default)s")
    p.add_argument(
        "--max_grad_norm", type=float, default=5.0, help="Threshold for gradient clipping. Default=%(default)s"
    )
    p.add_argument(
        "--iteration_per_update",
        type=int,
        default=1,
        help="Number of feed-forward iterations for one parameter update. Default=%(default)s",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="Initial learning rate. Default=%(default)s",
    )
    p.add_argument(
        "--lr_step",
        type=int,
        default=1,
        help="Number of epochs for each learning rate decay. Default=%(default)s",
    )
    p.add_argument(
        "--lr_gamma",
        type=float,
        default=0.5,
        help="Learning rate decay rate. Default=%(default)s",
    )
    p.add_argument(
        "--lr_decay_start",
        type=int,
        default=10,
        help="Learning rate decay start at. Default=%(default)s",
    )
    p.add_argument(
        "--use_adam",
        action="store_true",
        help="Use Adam as optimizer instead of SGD. Other lr arguments should be changed.",
    )
    p.add_argument(
        "--use_radam",
        action="store_true",
        help="Use rectified Adam as optimizer. Other lr arguments should be changed.",
    )
    p.add_argument(
        "--rl_lr", type=float, default=0.01, help="Learning rate for reinforcement learning. Default=%(default)s"
    )
    p.add_argument("--rl_n_samples", type=int, default=2, help="Number of samples to get baseline. Default=%(default)s")
    p.add_argument(
        "--rl_n_epochs", type=int, default=10, help="Number of epochs for reinforcement learning. Default=%(default)s"
    )
    p.add_argument(
        "--rl_n_gram",
        type=int,
        default=6,
        help="Maximum number of tokens to calculate BLEU for reinforcement learning. Default=%(default)s",
    )
    p.add_argument(
        "--rl_reward",
        type=str,
        default="gleu",
        help="Method name to use as reward function for RL training. Default=%(default)s",
    )
    p.add_argument(
        "--use_transformer",
        action="store_true",
        help="Set model architecture as Transformer.",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=8,
        help="Number of heads in multi-head attention in Transformer. Default=%(default)s",
    )
    config = p.parse_args()
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHON_HASH_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def read_data(file_path, src_tgt):
    data = pd.read_pickle(file_path)
    src_lang, tgt_lang = src_tgt[:2], src_tgt[2:]

    # parse source column and target column
    src_col, tgt_col = ("tok" + "_" + src_lang, "tok" + "_" + tgt_lang)
    srcs = data[src_col].tolist()
    tgts = data[tgt_col].tolist()
    return srcs, tgts


def get_loaders(config):
    # Get list of srcs and tgts
    train_srcs, train_tgts = read_data(config.train_fn, config.src_tgt)
    valid_srcs, valid_tgts = read_data(config.valid_fn, config.src_tgt)

    train_loader = DataLoader(
        TranslationDataset(train_srcs, train_tgts, with_text=config.with_text),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TranslationCollator(
            pad_idx=Vocabulary.PAD, max_length=config.max_length, with_text=config.with_text
        ),
    )

    train_src_vocab = train_loader.dataset.src_vocab
    train_tgt_vocab = train_loader.dataset.tgt_vocab

    valid_loader = DataLoader(
        TranslationDataset(
            valid_srcs, valid_tgts, src_vocab=train_src_vocab, tgt_vocab=train_tgt_vocab, with_text=config.with_text
        ),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TranslationCollator(
            pad_idx=Vocabulary.PAD, max_length=config.max_length, with_text=config.with_text
        ),
    )

    return train_loader, valid_loader


def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,
            config.hidden_size,
            output_size,
            n_splits=config.n_splits,
            n_enc_blocks=config.n_layers,
            n_dec_blocks=config.n_layers,
            dropout_p=config.dropout,
        )
    else:
        model = Seq2Seq(
            input_size,
            config.word_vec_size,
            config.hidden_size,
            output_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )

    return model


def get_crit(output_size, pad_idx):
    """
    Instead of using Cross-Entropy loss,
    we can use Negative Log-Likelihood(NLL) loss with log-probability.

    [TODO] Try using ignore_index = pad_idx
    https://discuss.pytorch.org/t/ignore-index-in-the-cross-entropy-loss/25006
    # crit = nn.NLLLoss(
    #     reduction = "sum",
    #     ignore_index = pad_idx
    # )
    """
    loss_weight = torch.ones(output_size)
    loss_weight[pad_idx] = 0.0

    crit = nn.NLLLoss(weight=loss_weight, reduction="sum")
    return crit


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98))
        else:  # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer


def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                i
                for i in range(
                    max(0, config.lr_decay_start - 1), (config.init_epoch - 1) + config.n_epochs, config.lr_step
                )
            ],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter()
        pp.pprint(vars(config))

    print_config(config)

    seed_everything(42)

    train_loader, valid_loader = get_loaders(config)
    input_size, output_size = (len(train_loader.dataset.src_vocab), len(train_loader.dataset.tgt_vocab))

    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, pad_idx=Vocabulary.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)

    if config.gpu_id >= 0 and not config.use_mps:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)
    elif config.use_mps:
        if torch.backends.mps.is_available():
            model.to("mps:{}".format(config.gpu_id))
            crit.to("mps:{}".format(config.gpu_id))
        else:
            raise NotImplementedError("No usable cuda or mps.")
    else:
        raise NotImplementedError("Current model cannot be trained with CPU. Please specify gpu device (cuda or mps).")

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        log.info(model)
        log.info(crit)
        log.info(optimizer)

    # Start training.
    mle_trainer = Trainer(MLEEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        src_vocab=train_loader.dataset.src_vocab,
        tgt_vocab=train_loader.dataset.tgt_vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler,
    )

    # Start RL trainig.
    if config.rl_n_epochs > 0:
        log.info("Start RL training.")
        optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)
        mrt_trainer = Trainer(MRTEngine, config)

        mrt_trainer.train(
            model,
            crit=None,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=train_loader.dataset.src_vocab,
            tgt_vocab=train_loader.dataset.tgt_vocab,
            n_epochs=config.rl_n_epochs,
            lr_scheduler=lr_scheduler,
        )


if __name__ == "__main__":
    config = define_argparser(is_continue=False)
    main(config)
