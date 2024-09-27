from chameleon.ignite_engine import LanguageModelTrainingEngine
from ignite.engine import Events


class LanguageModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(
        self, model, crit, optimizer, train_loader, valid_loader, src_vocab, tgt_vocab, n_epochs, lr_scheduler=None
    ):
        if src_vocab is not None and tgt_vocab is not None:
            raise NotImplementedError("You should assign None one of vocab to designate target language.")
        if src_vocab is None:
            is_src_target = False
        elif tgt_vocab is None:
            is_src_target = True
        else:
            raise NotImplementedError("You cannot assign None both vocab.")

        trainer = LanguageModelTrainingEngine(
            LanguageModelTrainingEngine.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            is_src_target,
            self.config,
        )
        evaluator = LanguageModelTrainingEngine(
            LanguageModelTrainingEngine.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            is_src_target=is_src_target,
            config=self.config,
        )

        LanguageModelTrainingEngine.attach_metrics(trainer, evaluator, verbose=self.config.verbose)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        trainer.add_event_handler(Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, LanguageModelTrainingEngine.check_best)
        # Pass
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            LanguageModelTrainingEngine.save_model,
            trainer,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        trainer.run(train_loader, max_epochs=n_epochs)

        if n_epochs > 0:
            model.load_state_dict(evaluator.best_model)

        return model
