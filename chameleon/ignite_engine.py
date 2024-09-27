import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from chameleon.base_dataset import Vocabulary
from chameleon.utils import get_grad_norm, get_parameter_norm

# torch
import torch
import torch.nn.utils as torch_utils
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# torch ignite
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

# reinforcement learning
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# logging
from loguru import logger as log
from chameleon.log import set_logger

set_logger(source="Chameleon", diagnose=True)


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

# for dual learning
X2Y, Y2X = 0, 1


class BaseEngine(Engine, metaclass=ABCMeta):
    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)  # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @abstractmethod
    def train(engine, mini_batch):
        pass

    @abstractmethod
    def validate(engine, mini_batch):
        pass

    @abstractmethod
    def attach_metrics(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        pass

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = resume_epoch - 1

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics["loss"]
        avg_valid_loss = engine.state.metrics["loss"]

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split(".")

        model_fn = (
            model_fn[:-1]
            + [
                "%02d" % train_engine.state.epoch,
                "%.2f-%.2f" % (avg_train_loss, np.exp(avg_train_loss)),
                "%.2f-%.2f" % (avg_valid_loss, np.exp(avg_valid_loss)),
            ]
            + [model_fn[-1]]
        )

        model_fn = ".".join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                "model": engine.model.state_dict(),
                "opt": train_engine.optimizer.state_dict(),
                "config": config,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
            },
            model_fn,
        )


class MLEEngine(BaseEngine):
    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.lr_scheduler = lr_scheduler
        self.scaler = GradScaler()

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
        mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

        # Raw target variable has both BOS and EOS token.
        # The output of sequence-to-sequence does not have BOS token.
        # Thus, remove BOS token for reference.
        x, y = mini_batch["input_ids"], mini_batch["output_ids"][0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        with autocast(not engine.config.off_autocast):
            # Take feed-forward
            # Similar as before, the input of decoder does not have EOS token.
            # Thus, remove EOS token for decoder input.
            y_hat = engine.model(x, mini_batch["output_ids"][0][:, :-1])
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1))

            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = int(mini_batch["output_ids"][1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and engine.state.iteration > 0:
            # In orter to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            # Take a step of gradient descent.
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                # Use scaler instead of engine.optimizer.step() if using GPU.
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            "loss": loss,
            "ppl": ppl,
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.0,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.0,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
            mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

            x, y = mini_batch["input_ids"], mini_batch["output_ids"][0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch["output_ids"][0][:, :-1])
                # |y_hat| = (batch_size, n_classes)
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                )

        word_count = int(mini_batch["output_ids"][1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            "loss": loss,
            "ppl": ppl,
        }

    @staticmethod
    def attach_metrics(
        train_engine,
        validation_engine,
        training_metric_names=["loss", "ppl", "|param|", "|g_param|"],
        validation_metric_names=["loss", "ppl"],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:

            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics["|param|"]
                avg_g_norm = engine.state.metrics["|g_param|"]
                avg_loss = engine.state.metrics["loss"]

                log.info(
                    "Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}".format(
                        engine.state.epoch,
                        avg_p_norm,
                        avg_g_norm,
                        avg_loss,
                        np.exp(avg_loss),
                    )
                )

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        # Check best loss at the end of the epoch of the validation_engine
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            MLEEngine.check_best,  # function
        )

        if verbose >= VERBOSE_EPOCH_WISE:

            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics["loss"]

                log.info(
                    "Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}".format(
                        avg_loss,
                        np.exp(avg_loss),
                        engine.best_loss,
                        np.exp(engine.best_loss),
                    )
                )


class MRTEngine(BaseEngine):
    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        # Unlike MLEEngine, it uses another type of criterion when calculating loss.
        # To get loss in MRTEngine, it needs reward.
        self.lr_scheduler = lr_scheduler
        self.scaler = GradScaler()

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def _get_reward(y_hat, y, n_gram=6, method="gleu"):
        # |y| = (batch_size, length)
        # |y_hat| = (batch_size, length)
        # |scores(output)| = (batch_size, )

        # y_hat and y are all one-hot vectors.
        # GLEU is a type of variation from BLEU, which was developeed by Google.
        # It is more fit to reinforcement learning.
        # About SmoothingFunction of BLEU, please refer to the documentation.
        sf = SmoothingFunction()
        score_func = {
            "gleu": lambda ref, hyp: sentence_gleu([ref], hyp, max_len=n_gram),
            "bleu1": lambda ref, hyp: sentence_bleu(
                [ref], hyp, weights=[1.0 / n_gram] * n_gram, smoothing_function=sf.method1
            ),
            "bleu2": lambda ref, hyp: sentence_bleu(
                [ref], hyp, weights=[1.0 / n_gram] * n_gram, smoothing_function=sf.method2
            ),
            "bleu3": lambda ref, hyp: sentence_bleu(
                [ref], hyp, weights=[1.0 / n_gram] * n_gram, smoothing_function=sf.method3
            ),
        }[method]

        with torch.no_grad():
            scores = []

            for sample_idx in range(y.size(0)):
                # ref = y[sample_idx].masked_select(y[sample_idx] != Vocabulary.EOS).tolist()
                # hyp = y_hat[sample_idx].masked_select(y_hat[sample_idx] != Vocabulary.EOS).tolist()

                ref, hyp = [], []

                for time_step in range(y.size(-1)):
                    ref += [str(int(y[sample_idx, time_step]))]
                    if y[sample_idx, time_step] == Vocabulary.EOS:
                        break

                for time_step in range(y_hat.size(-1)):
                    hyp += [str(int(y_hat[sample_idx, time_step]))]
                    if y_hat[sample_idx, time_step] == Vocabulary.EOS:
                        break

                scores += [score_func(ref, hyp) * 100.0]

            scores = torch.FloatTensor(scores).to(y.device)
            # |scores| = (batch_size, )
            return scores

    @staticmethod
    def _get_loss(y_hat, indice, reward):
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)
        # |reward| = (batch_size, )

        # Unlike MLEEngine, it does not calculate the log-likelihood.
        # Instead, it calculates the log-probability from the policy.

        batch_size = y_hat.size(0)
        output_size = y_hat.size(-1)

        # #[TODO] Intuitive, but memory inefficient
        # mask = indice == Vocabulary.PAD
        # # |mask| = (batch_size, length)
        # indice = F.one_hot(indice, num_classes=output_size).float()
        # # |indice| = (batch_size, length, output_size)
        # log_prob = (y_hat * indice).sum(dim=-1)
        # # |log_prob| = (batch_size, length)
        # log_prob.masked_fill_(mask, 0)
        # log_prob.sum(dim=-1)
        # # |log_prob| = (batch_size)

        # memory effiecient
        log_prob = (
            -F.nll_loss(y_hat.view(-1, output_size), indice.view(-1), ignore_index=Vocabulary.PAD, reduction="none")
            .view(batch_size, -1)
            .sum(dim=-1)
        )

        loss = (log_prob * -reward).sum()
        return loss

    @staticmethod
    def train(engine, mini_batch):
        # Refer to below about backprop through Random Sampling.
        # https://ai.stackexchange.com/questions/33824/how-does-backprop-work-through-the-random-sampling-layer-in-a-variational-autoen

        engine.model.train()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
        mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

        x, y = mini_batch["input_ids"], mini_batch["output_ids"][0][:, 1:]
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # Take sampling process by setting is_greedy as False
        y_hat, indice = engine.model.search(x, is_greedy=False, max_length=engine.config.max_length)
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        with torch.no_grad():
            # Based on the result of sampling, get reward
            actor_reward = MRTEngine._get_reward(
                indice, y, n_gram=engine.config.rl_n_gram, method=engine.config.rl_reward
            )
            # |actor_reward| = (batch_size, )

            # Get state value function (baseline)
            # Here does not train another neural network to get V(s)
            # Instead, take samples as many as n_samples, and get average rewards.
            baseline = []

            for _ in range(engine.config.rl_n_samples):
                _, sampled_indice = engine.model.search(x, is_greedy=False, max_length=engine.config.max_length)
                baseline += [
                    MRTEngine._get_reward(
                        sampled_indice, y, n_gram=engine.config.rl_n_gram, method=engine.config.rl_reward
                    )
                ]
            # |baseline| = (n_samples, batch_size) # dtype=list
            baseline = torch.stack(baseline).mean(dim=0)
            # |baseline| = (batch_size, )

            reward = actor_reward - baseline
            # |reward| = (batch_size, )

        # calculate gradients with back propagation
        with autocast(not engine.config.off_autocast):
            loss = MRTEngine._get_loss(y_hat, indice, reward=reward)
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
            backward_target.backward()

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and engine.state.iteration > 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
            else:
                engine.optimizer.step()

        return {
            "actor": float(actor_reward.mean()),
            "baseline": float(baseline.mean()),
            "reward": float(reward.mean()),
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.0,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.0,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
            mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

            x, y = mini_batch["input_ids"], mini_batch["output_ids"][0][:, 1:]
            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            # Take sampling process by setting is_greedy as False
            y_hat, indice = engine.model.search(x, is_greedy=False, max_length=engine.config.max_length)
            # |y_hat| = (batch_size, length, output_size)
            # |indice| = (batch_size, length)

            # Based on the result of sampling, get reward
            actor_reward = MRTEngine._get_reward(
                indice, y, n_gram=engine.config.rl_n_gram, method=engine.config.rl_reward
            )
            # |actor_reward| = (batch_size, )

        return {"BLEU": float(actor_reward.mean())}

    @staticmethod
    def check_best(engine):
        loss = -float(engine.state.metrics["BLEU"])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_bleu = train_engine.state.metrics["actor"]
        avg_valid_bleu = engine.state.metrics["BLEU"]

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split(".")

        model_fn = (
            model_fn[:-1]
            + [
                "mrt",
                "%02d" % train_engine.state.epoch,
                "%.2f-%.2f" % (avg_train_bleu, avg_valid_bleu),
            ]
            + [model_fn[-1]]
        )

        model_fn = ".".join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                "model": engine.model.state_dict(),
                "opt": train_engine.optimizer.state_dict(),
                "config": config,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
            },
            model_fn,
        )

    @staticmethod
    def attach_metrics(
        train_engine,
        validation_engine,
        training_metric_names=["actor", "baseline", "reward", "|param|", "|g_param|"],
        validation_metric_names=["BLEU"],
        verbose=VERBOSE_BATCH_WISE,
    ):
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(engine, metric_name)

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:

            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics["|param|"]
                avg_g_norm = engine.state.metrics["|g_param|"]
                avg_reward = engine.state.metrics["actor"]

                log.info(
                    "Epoch {} - |param|={:.2e} |g_param|={:.2e} BLEU={:.4e}".format(
                        engine.state.epoch,
                        avg_p_norm,
                        avg_g_norm,
                        avg_reward,
                    )
                )

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        # Check best loss at the end of the epoch of the validation_engine
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            MRTEngine.check_best,  # function
        )

        if verbose >= VERBOSE_EPOCH_WISE:

            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_bleu = engine.state.metrics["BLEU"]
                log.info(
                    "Validation - BLEU={:.2f} best_BLEU={:.2f}".format(
                        avg_bleu,
                        -engine.best_loss,
                    )
                )


class LanguageModelTrainingEngine(MLEEngine):
    def __init__(self, func, model, crit, optimizer, lr_scheduler, is_src_target, config):
        self.is_src_target = is_src_target

        super().__init__(func, model, crit, optimizer, lr_scheduler, config)

        self.best_model = None
        self.scaler = GradScaler()

    @staticmethod
    def train(engine, mini_batch):
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train()
        engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
        mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

        # if 'is_src_target' is true, the trainer would train language model for source language.
        # For dsl case, both x and y has BOS and EOS tokens.
        # Thus, we need to remove BOS and EOS before the training.
        x = mini_batch["input_ids"][0][:, :-1] if engine.is_src_target else mini_batch["output_ids"][0][:, :-1]
        y = mini_batch["input_ids"][0][:, 1:] if engine.is_src_target else mini_batch["output_ids"][0][:, 1:]
        # |x| = |y| = (batch_size, length)

        with autocast(not engine.config.off_autocast):
            y_hat = engine.model(x)
            # |y_hat| = (batch_size, length, output_size)

            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            ).sum()
            backward_target = loss.div(y.size(0))

        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()

        word_count = (
            int(mini_batch["input_ids"][1].sum()) if engine.is_src_target else int(mini_batch["output_ids"][1].sum())
        )
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # In orther to avoid gradient exploding, we apply gradient clipping.
        torch_utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        # Take a step of gradient descent.
        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            # Use scaler instead of engine.optimizer.step() if using GPU.
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            "loss": loss,
            "ppl": ppl,
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.0,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.0,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1])
            mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1])

            x = mini_batch["input_ids"][0][:, :-1] if engine.is_src_target else mini_batch["output_ids"][0][:, :-1]
            y = mini_batch["input_ids"][0][:, 1:] if engine.is_src_target else mini_batch["output_ids"][0][:, 1:]
            # |x| = |y| = (batch_size, length)

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x)
                # |y_hat| = (batch_size, length, output_size)

                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1),
                ).sum()

        word_count = (
            int(mini_batch["input_ids"][1].sum()) if engine.is_src_target else int(mini_batch["output_ids"][1].sum())
        )
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            "loss": loss,
            "ppl": ppl,
        }

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        pass


class DSLEngine(Engine):
    def __init__(self, func, models, crits, optimizers, lr_schedulers, language_models, config):
        self.models = models
        self.crits = crits
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.language_models = language_models
        self.config = config

        super().__init__(func)

        self.best_x2y = np.inf
        self.best_y2x = np.inf
        self.scalers = [
            GradScaler(),
            GradScaler(),
        ]

    @staticmethod
    def _reorder(x, y, l):
        # This method is one of important methods in this class.
        # Since encoder takes packed_sequence instance,
        # the samples in mini-batch must be sorted by lengths.
        # Thus, we need to re-order the samples in mini-batch, if src and tgt is reversed.
        # (Because originally src and tgt are sorted by the length of samples in src.)

        # sort by length.
        indice = l.sort(descending=True)[1]

        # re-order based on the indice.
        x_ = x.index_select(dim=0, index=indice).contiguous()
        y_ = y.index_select(dim=0, index=indice).contiguous()
        l_ = l.index_select(dim=0, index=indice).contiguous()

        # generate information to restore the re-ordering.
        restore_indice = indice.sort(descending=False)[1]

        return x_, (y_, l_), restore_indice

    @staticmethod
    def _restore_order(x, restore_indice):
        return x.index_select(dim=0, index=restore_indice)

    @staticmethod
    def _get_loss(x, y, x_hat, y_hat, crits, x_lm=None, y_lm=None, lagrange=1e-3):
        # |x| = (batch_size, n)
        # |y| = (batch_size, m)
        # |x_hat| = (batch_size, n, output_size0)
        # |y_hat| = (batch_size, m, output_size1)
        # |x_lm| = |x_hat|
        # |y_lm| = |y_hat|

        log_p_y_given_x = -crits[X2Y](
            y_hat.contiguous().view(-1, y_hat.size(-1)),
            y.contiguous().view(-1),
        )
        log_p_x_given_y = -crits[Y2X](
            x_hat.contiguous().view(-1, x_hat.size(-1)),
            x.contiguous().view(-1),
        )
        # |log_p_y_given_x| = (batch_size * m)
        # |log_p_x_given_y| = (batch_size * n)

        log_p_y_given_x = log_p_y_given_x.view(y.size(0), -1).sum(dim=-1)
        log_p_x_given_y = log_p_x_given_y.view(x.size(0), -1).sum(dim=-1)
        # |log_p_y_given_x| = |log_p_x_given_y| = (batch_size, )

        # Negative Log-likelihood
        loss_x2y = -log_p_y_given_x
        loss_y2x = -log_p_x_given_y

        if x_lm is not None and y_lm is not None:
            log_p_x = -crits[Y2X](
                x_lm.contiguous().view(-1, x_lm.size(-1)),
                x.contiguous().view(-1),
            )
            log_p_y = -crits[X2Y](
                y_lm.contiguous().view(-1, y_lm.size(-1)),
                y.contiguous().view(-1),
            )
            # |log_p_x| = (batch_size * n)
            # |log_p_y| = (batch_size * m)

            log_p_x = log_p_x.view(x.size(0), -1).sum(dim=-1)
            log_p_y = log_p_y.view(y.size(0), -1).sum(dim=-1)
            # |log_p_x| = (batch_size, )
            # |log_p_y| = (batch_size, )

            # Just for logging: both losses are detached.
            dual_loss = lagrange * ((log_p_x + log_p_y_given_x.detach()) - (log_p_y + log_p_x_given_y.detach())) ** 2

            # Note that 'detach()' is used to prevent unnecessary back-propagation.
            loss_x2y += lagrange * ((log_p_x + log_p_y_given_x) - (log_p_y + log_p_x_given_y.detach())) ** 2
            loss_y2x += lagrange * ((log_p_x + log_p_y_given_x.detach()) - (log_p_y + log_p_x_given_y)) ** 2
        else:
            dual_loss = None

        return (
            loss_x2y.sum(),
            loss_y2x.sum(),
            float(dual_loss.sum()) if dual_loss is not None else 0.0,
        )

    @staticmethod
    def train(engine, mini_batch):
        for language_model, model, optimizer in zip(engine.language_models, engine.models, engine.optimizers):
            language_model.eval()
            model.train()
            if (
                engine.state.iteration % engine.config.iteration_per_update == 1
                or engine.config.iteration_per_update == 1
            ):
                if engine.state.iteration > 1:
                    optimizer.zero_grad()

        device = next(engine.models[0].parameters()).device
        mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1].to(device))
        mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1].to(device))

        with autocast(not engine.config.off_autocast):
            # X2Y
            x, y = (mini_batch["input_ids"][0][:, 1:-1], mini_batch["input_ids"][1] - 2), mini_batch["output_ids"][0][
                :, :-1
            ]
            x_hat_lm, y_hat_lm = None, None
            # |x| = (batch_size, n)
            # |y| = (batch_size, m)
            y_hat = engine.models[X2Y](x, y)
            # |y_hat| = (batch_size, m, y_vocab_size)

            if engine.state.epoch > engine.config.dsl_n_warmup_epochs:
                with torch.no_grad():
                    y_hat_lm = engine.language_models[X2Y](y)
                    # |y_hat_lm| = |y_hat|

            # Y2X
            # Since encoder in seq2seq takes packed_sequence instance,
            # we need to re-sort if we use reversed src and tgt.
            x, y, restore_indice = DSLEngine._reorder(
                mini_batch["input_ids"][0][:, :-1],
                mini_batch["output_ids"][0][:, 1:-1],
                mini_batch["output_ids"][1] - 2,
            )
            # |x| = (batch_size, n)
            # |y| = (batch_size, m)
            x_hat = DSLEngine._restore_order(
                engine.models[Y2X](y, x),
                restore_indice=restore_indice,
            )
            # |x_hat| = (batch_size, n, x_vocab_size)

            if engine.state.epoch > engine.config.dsl_n_warmup_epochs:
                with torch.no_grad():
                    x_hat_lm = DSLEngine._restore_order(
                        engine.language_models[Y2X](x),
                        restore_indice=restore_indice,
                    )
                    # |x_hat_lm| = |x_hat|

            x, y = mini_batch["input_ids"][0][:, 1:], mini_batch["output_ids"][0][:, 1:]
            loss_x2y, loss_y2x, dual_loss = DSLEngine._get_loss(
                x,
                y,
                x_hat,
                y_hat,
                engine.crits,
                x_hat_lm,
                y_hat_lm,
                # According to the paper, DSL should be warm-started.
                # Thus, we turn-off the regularization at the beginning.
                lagrange=engine.config.dsl_lambda if engine.state.epoch > engine.config.dsl_n_warmup_epochs else 0.0,
            )

            backward_targets = [
                loss_x2y.div(y.size(0)).div(engine.config.iteration_per_update),
                loss_y2x.div(x.size(0)).div(engine.config.iteration_per_update),
            ]

        for scaler, backward_target in zip(engine.scalers, backward_targets):
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                scaler.scale(backward_target).backward()
            else:
                backward_target.backward()

        x_word_count = int(mini_batch["input_ids"][1].sum())
        y_word_count = int(mini_batch["output_ids"][1].sum())
        p_norm = float(
            get_parameter_norm(list(engine.models[X2Y].parameters()) + list(engine.models[Y2X].parameters()))
        )
        g_norm = float(get_grad_norm(list(engine.models[X2Y].parameters()) + list(engine.models[Y2X].parameters())))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and engine.state.iteration > 0:
            for model, optimizer, scaler in zip(engine.models, engine.optimizers, engine.scalers):
                torch_utils.clip_grad_norm_(
                    model.parameters(),
                    engine.config.max_grad_norm,
                )
                # Take a step of gradient descent.
                if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                    # Use scaler instead of engine.optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        return {
            "x2y": float(loss_x2y / y_word_count),
            "y2x": float(loss_y2x / x_word_count),
            "reg": float(dual_loss / x.size(0)),
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.0,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.0,
        }

    @staticmethod
    def validate(engine, mini_batch):
        for model in engine.models:
            model.eval()

        with torch.no_grad():
            device = next(engine.models[0].parameters()).device
            mini_batch["input_ids"] = (mini_batch["input_ids"][0].to(device), mini_batch["input_ids"][1].to(device))
            mini_batch["output_ids"] = (mini_batch["output_ids"][0].to(device), mini_batch["output_ids"][1].to(device))

            with autocast(not engine.config.off_autocast):
                # X2Y
                x, y = (mini_batch["input_ids"][0][:, 1:-1], mini_batch["input_ids"][1] - 2), mini_batch["output_ids"][
                    0
                ][:, :-1]
                # |x| = (batch_size, n)
                # |y| = (batch_size  m)
                y_hat = engine.models[X2Y](x, y)
                # |y_hat| = (batch_size, m, y_vocab_size)

                # Y2X
                x, y, restore_indice = DSLEngine._reorder(
                    mini_batch["input_ids"][0][:, :-1],
                    mini_batch["output_ids"][0][:, 1:-1],
                    mini_batch["output_ids"][1] - 2,
                )
                x_hat = DSLEngine._restore_order(
                    engine.models[Y2X](y, x),
                    restore_indice=restore_indice,
                )
                # |x_hat| = (batch_size, n, x_vocab_size)

                # You don't have to use _get_loss method,
                # because we don't have to care about the gradients.
                x, y = mini_batch["input_ids"][0][:, 1:], mini_batch["output_ids"][0][:, 1:]
                loss_x2y = engine.crits[X2Y](y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1)).sum()
                loss_y2x = engine.crits[Y2X](x_hat.contiguous().view(-1, x_hat.size(-1)), x.contiguous().view(-1)).sum()

                x_word_count = int(mini_batch["input_ids"][1].sum())
                y_word_count = int(mini_batch["output_ids"][1].sum())

        return {
            "x2y": float(loss_x2y / y_word_count),
            "y2x": float(loss_y2x / x_word_count),
        }

    @staticmethod
    def attach_metrics(
        train_engine,
        validation_engine,
        training_metric_names=["x2y", "y2x", "reg", "|param|", "|g_param|"],
        validation_metric_names=["x2y", "y2x"],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:

            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics["|param|"]
                avg_g_norm = engine.state.metrics["|g_param|"]
                avg_x2y = engine.state.metrics["x2y"]
                avg_y2x = engine.state.metrics["y2x"]
                avg_reg = engine.state.metrics["reg"]

                print(
                    """Epoch {} - |param|={:.2e} |g_param|={:.2e}
                    loss_x2y={:.4e} ppl_x2y={:.2f} loss_y2x={:.4e}
                    ppl_y2x={:.2f} dual_loss={:.4e}""".format(
                        engine.state.epoch,
                        avg_p_norm,
                        avg_g_norm,
                        avg_x2y,
                        np.exp(avg_x2y),
                        avg_y2x,
                        np.exp(avg_y2x),
                        avg_reg,
                    )
                )

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        # Check best loss at the end of the epoch of the validation_engine
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # event
            DSLEngine.check_best,  # function
        )

        if verbose >= VERBOSE_EPOCH_WISE:

            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_x2y = engine.state.metrics["x2y"]
                avg_y2x = engine.state.metrics["y2x"]

                print(
                    "Validation X2Y - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}".format(
                        avg_x2y,
                        np.exp(avg_x2y),
                        engine.best_x2y,
                        np.exp(engine.best_x2y),
                    )
                )
                print(
                    "Validation Y2X - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}".format(
                        avg_y2x,
                        np.exp(avg_y2x),
                        engine.best_y2x,
                        np.exp(engine.best_y2x),
                    )
                )

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = resume_epoch - 1

    @staticmethod
    def check_best(engine):
        x2y = float(engine.state.metrics["x2y"])
        if x2y <= engine.best_x2y:
            engine.best_x2y = x2y
        y2x = float(engine.state.metrics["y2x"])
        if y2x <= engine.best_y2x:
            engine.best_y2x = y2x

    @staticmethod
    def save_model(engine, train_engine, config, vocabs):
        avg_train_x2y = train_engine.state.metrics["x2y"]
        avg_train_y2x = train_engine.state.metrics["y2x"]
        avg_valid_x2y = engine.state.metrics["x2y"]
        avg_valid_y2x = engine.state.metrics["y2x"]

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split(".")

        model_fn = (
            model_fn[:-1]
            + [
                "%02d" % train_engine.state.epoch,
                "%.2f-%.2f" % (avg_train_x2y, np.exp(avg_train_x2y)),
                "%.2f-%.2f" % (avg_train_y2x, np.exp(avg_train_y2x)),
                "%.2f-%.2f" % (avg_valid_x2y, np.exp(avg_valid_x2y)),
                "%.2f-%.2f" % (avg_valid_y2x, np.exp(avg_valid_y2x)),
            ]
            + [model_fn[-1]]
        )

        model_fn = ".".join(model_fn)

        torch.save(
            {
                "model": [
                    train_engine.models[0].state_dict(),
                    train_engine.models[1].state_dict(),
                    train_engine.language_models[0].state_dict(),
                    train_engine.language_models[1].state_dict(),
                ],
                "opt": [
                    train_engine.optimizers[0].state_dict(),
                    train_engine.optimizers[1].state_dict(),
                ],
                "config": config,
                "src_vocab": vocabs[0],
                "tgt_vocab": vocabs[1],
            },
            model_fn,
        )
