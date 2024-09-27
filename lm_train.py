import argparse
import pprint
import torch
import torch.nn as nn
import pandas as pd

from chameleon.dl_dataset import Vocabulary, TranslationDataset, TranslationCollator
from torch.utils.data import DataLoader
from torch import optim
from chameleon.models.lstm_lm import LanguageModel
from chameleon.lm_trainer import LanguageModelTrainer
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
    p.add_argument(
        "--train_fn",
        required=not is_continue,
        help="Training set file name except the extention. (ex: train.en --> train)",
    )
    p.add_argument(
        "--valid_fn",
        required=not is_continue,
        help="Validation set file name except the extention. (ex: valid.en --> valid)",
    )
    p.add_argument(
        "--tgt_src",
        default="enko",
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
        "--max_length", type=int, default=100, help="Maximum length of the training sequence. Default=%(default)s"
    )
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate. Default=%(default)s")
    p.add_argument(
        "--word_vec_size", type=int, default=512, help="Word embedding vector dimension. Default=%(default)s"
    )
    p.add_argument("--hidden_size", type=int, default=768, help="Hidden size of LSTM. Default=%(default)s")
    p.add_argument("--n_layers", type=int, default=4, help="Number of layers in LSTM. Default=%(default)s")
    p.add_argument(
        "--max_grad_norm", type=float, default=1e8, help="Threshold for gradient clipping. Default=%(default)s"
    )

    config = p.parse_args()

    return config


def get_models(src_vocab_size, tgt_vocab_size, config):
    language_models = [
        LanguageModel(
            tgt_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout=config.dropout,
        ),
        LanguageModel(
            src_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout=config.dropout,
        ),
    ]

    return language_models


def read_data(file_path, tgt_src):
    data = pd.read_pickle(file_path)
    tgt_lang, src_lang = tgt_src[:2], tgt_src[2:]

    # parse source column and target column
    tgt_col, src_col = ("tok" + "_" + tgt_lang, "tok" + "_" + src_lang)
    tgts = data[tgt_col].tolist()
    srcs = data[src_col].tolist()
    return tgts, srcs


def get_loaders(config):
    # Get list of srcs and tgts
    train_tgts, train_srcs = read_data(config.train_fn, config.tgt_src)
    valid_tgts, valid_srcs = read_data(config.valid_fn, config.tgt_src)

    log.info(f"Target Language: {config.tgt_src[:2]}, Source Language: {config.tgt_src[-2:]}")
    # DataLoader for source and target language
    train_loader = DataLoader(
        TranslationDataset(
            srcs=train_srcs,
            tgts=train_tgts,
            with_text=False,
            is_dual=True,  # tgt dataset also needs BOS and EOS token at the begging and the end.
        ),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TranslationCollator(
            pad_idx=Vocabulary.PAD, eos_idx=Vocabulary.EOS, max_length=config.max_length, with_text=False, is_dual=True
        ),
    )

    train_src_vocab = train_loader.dataset.src_vocab
    train_tgt_vocab = train_loader.dataset.tgt_vocab

    valid_loader = DataLoader(
        TranslationDataset(
            srcs=valid_srcs,
            tgts=valid_tgts,
            src_vocab=train_src_vocab,
            tgt_vocab=train_tgt_vocab,
            with_text=True,
            is_dual=True,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TranslationCollator(
            pad_idx=Vocabulary.PAD, eos_idx=Vocabulary.EOS, max_length=config.max_length, with_text=True, is_dual=True
        ),
    )

    return train_loader, valid_loader


def get_crits(src_vocab_size, tgt_vocab_size, pad_index):
    loss_weights = [
        torch.ones(tgt_vocab_size),
        torch.ones(src_vocab_size),
    ]
    loss_weights[0][pad_index] = 0.0
    loss_weights[1][pad_index] = 0.0

    crits = [
        nn.NLLLoss(weight=loss_weights[0], reduction="none"),
        nn.NLLLoss(weight=loss_weights[1], reduction="none"),
    ]

    return crits


def main(config):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    print_config(config)

    # get loader
    train_loader, valid_loader = get_loaders(config)

    src_vocab_size = len(train_loader.dataset.src_vocab)
    tgt_vocab_size = len(train_loader.dataset.tgt_vocab)

    models = get_models(src_vocab_size, tgt_vocab_size, config)

    crits = get_crits(src_vocab_size, tgt_vocab_size, pad_index=Vocabulary.PAD)

    if config.gpu_id >= 0 and not config.use_mps:
        for model, crit in zip(models, crits):
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    elif config.use_mps:
        for model, crit in zip(models, crits):
            model.to("mps:{}".format(config.gpu_id))
            crit.to("mps:{}".format(config.gpu_id))

    if config.verbose >= 2:
        print(models)

    for model, crit in zip(models, crits):
        optimizer = optim.Adam(model.parameters())
        lm_trainer = LanguageModelTrainer(config)

        model = lm_trainer.train(
            model,
            crit,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            src_vocab=train_loader.dataset.src_vocab if model.vocab_size == src_vocab_size else None,
            tgt_vocab=train_loader.dataset.tgt_vocab if model.vocab_size == tgt_vocab_size else None,
            n_epochs=config.n_epochs,
        )

    torch.save(
        {
            "model": [
                models[0].state_dict(),
                models[1].state_dict(),
            ],
            "config": config,
            "src_vocab": train_loader.dataset.src_vocab,
            "tgt_vocab": train_loader.dataset.tgt_vocab,
        },
        config.model_fn,
    )


if __name__ == "__main__":
    config = define_argparser(is_continue=False)
    main(config)
