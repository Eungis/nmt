import argparse
import pprint
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chameleon.dl_dataset import Vocabulary, TranslationDataset, TranslationCollator
from chameleon.models.seq2seq import Seq2Seq
from chameleon.models.transformer import Transformer
from chameleon.models.lstm_lm import LanguageModel
from chameleon.dual_trainer import DSLTrainer
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
    p.add_argument("--lm_fn", required=not is_continue, help="LM file name, which is trained with lm_train.py.")
    p.add_argument(
        "--train",
        required=not is_continue,
        help="Training set file name except the extention. (ex: train.en --> train)",
    )
    p.add_argument(
        "--valid",
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
        "--init_epoch",
        required=is_continue,
        type=int,
        default=1,
        help="Set initial epoch number, which can be useful in continue training. Default=%(default)s",
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
    p.add_argument(
        "--iteration_per_update",
        type=int,
        default=1,
        help="Number of feed-forward iterations for one parameter update. Default=%(default)s",
    )

    p.add_argument(
        "--dsl_n_warmup_epochs",
        type=int,
        default=2,
        help="Number of warmup epochs for Dual Supervised Learning. Default=%(default)s",
    )
    p.add_argument(
        "--dsl_lambda",
        type=float,
        default=1e-3,
        help="Lagrangian Multiplier for regularization term. Default=%(default)s",
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


def load_lm(fn, language_models):
    saved_data = torch.load(fn, map_location="cpu")

    model_weight = saved_data["model"]
    language_models[0].load_state_dict(model_weight[0])
    language_models[1].load_state_dict(model_weight[1])


def get_models(src_vocab_size, tgt_vocab_size, config):
    language_models = [
        LanguageModel(
            tgt_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
        LanguageModel(
            src_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
    ]

    if config.use_transformer:
        models = [
            Transformer(
                src_vocab_size,
                config.hidden_size,
                tgt_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout,
            ),
            Transformer(
                tgt_vocab_size,
                config.hidden_size,
                src_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]
    else:
        models = [
            Seq2Seq(
                src_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                tgt_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            Seq2Seq(
                tgt_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                src_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]

    return language_models, models


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


def get_optimizers(models, config):
    if config.use_transformer:
        optimizers = [
            optim.Adam(models[0].parameters(), betas=(0.9, 0.98)),
            optim.Adam(models[1].parameters(), betas=(0.9, 0.98)),
        ]
    else:
        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

    return optimizers


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    print_config(config)

    # get loader
    train_loader, valid_loader = get_loaders(config)

    src_vocab_size = len(train_loader.dataset.src_vocab)
    tgt_vocab_size = len(train_loader.dataset.tgt_vocab)

    language_models, models = get_models(src_vocab_size, tgt_vocab_size, config)

    crits = get_crits(src_vocab_size, tgt_vocab_size, pad_index=Vocabulary.PAD)

    if model_weight is not None:
        for model, w in zip(models + language_models, model_weight):
            model.load_state_dict(w)

    load_lm(config.lm_fn, language_models)

    if config.gpu_id >= 0:
        for lm, seq2seq, crit in zip(language_models, models, crits):
            lm.cuda(config.gpu_id)
            seq2seq.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    dsl_trainer = DSLTrainer(config)

    optimizers = get_optimizers(models, config)

    if opt_weight is not None:
        for opt, w in zip(optimizers, opt_weight):
            opt.load_state_dict(w)

    if config.verbose >= 2:
        print(language_models)
        print(models)
        print(crits)
        print(optimizers)

    dsl_trainer.train(
        models,
        language_models,
        crits,
        optimizers,
        train_loader=train_loader,
        valid_loader=valid_loader,
        vocabs=[train_loader.dataset.src_vocab, train_loader.dataset.tgt_vocab],
        n_epochs=config.n_epochs,
        lr_schedulers=None,
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
