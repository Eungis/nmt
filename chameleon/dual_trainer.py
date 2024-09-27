from ignite.engine import Events
from chameleon.ignite_engine import DSLEngine


class DSLTrainer:
    def __init__(self, config):
        self.config = config

    def train(
        self,
        models,
        language_models,
        crits,
        optimizers,
        train_loader,
        valid_loader,
        vocabs,
        n_epochs,
        lr_schedulers=None,
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = DSLEngine(
            DSLEngine.train,
            models,
            crits,
            optimizers,
            lr_schedulers,
            language_models,
            self.config,
        )
        validation_engine = DSLEngine(
            DSLEngine.validate,
            models,
            crits,
            optimizers=None,
            lr_schedulers=None,
            language_models=language_models,
            config=self.config,
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        DSLEngine.attach_metrics(train_engine, validation_engine, verbose=self.config.verbose)

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_schedulers is not None:
                for s in engine.lr_schedulers:
                    s.step()

        # Attach above call-back function.
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_validation, validation_engine, valid_loader)
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            DSLEngine.resume_training,
            self.config.init_epoch,
        )

        # Attach validation loss check procedure for every end of validation epoch.
        validation_engine.add_event_handler(Events.EPOCH_COMPLETED, DSLEngine.check_best)
        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            DSLEngine.save_model,
            train_engine,
            self.config,
            vocabs,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return models
