from ignite.engine import Events


class Trainer:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def train(
        self, model, crit, optimizer, train_loader, valid_loader, src_vocab, tgt_vocab, n_epochs, lr_scheduler=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = self.engine(self.engine.train, model, crit, optimizer, lr_scheduler, self.config)
        validation_engine = self.engine(
            self.engine.validate, model, crit, optimizer=None, lr_scheduler=None, config=self.config
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        self.engine.attach_metrics(train_engine, validation_engine, verbose=self.config.verbose)

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        # Attach above call-back function.
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_validation, validation_engine, valid_loader)

        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            self.engine.resume_training,
            self.config.init_epoch,
        )

        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.engine.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model
