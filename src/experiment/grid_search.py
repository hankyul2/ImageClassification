import os
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from src import base_model_wrapper
from src import log


def change_model_wrapper(c):
    c.__init__ = lambda *args, **kwargs: None
    c.log = lambda x, y: None
    c.log_tensorboard = lambda a, b, c, d, e, f: None
    c.save_best_weight = lambda x, y, w, z: None
    c.valid = change_valid(c.valid)


def change_valid(fn):
    def wrapper(*args, **kwargs):
        loss, accuracy = fn(*args, **kwargs)
        tune.report(loss=loss, accuracy=accuracy)
        return loss, accuracy
    return wrapper


def change_result(c):
    c.__init__ = partial(c.__init__, result_path='log/grid_search.csv')


def apply_search_wrapper():
    change_model_wrapper(base_model_wrapper.BaseModelWrapper)
    change_result(log.Result)


def change_main(main):
    def wrapper(config, checkpoint_dir=None, data_dir=None):
        args = config['args']
        args.lr = config['lr']
        args.batch_size = config['batch_size']
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../')
        out = main(args)
        return out
    return wrapper


def get_utils(args, max_epoch):
    config = {
        'args': args,
        'lr': tune.loguniform(3e-4, 3e-2),
        'batch_size': tune.choice(list(range(16, 32)))
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epoch,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    return config, scheduler, reporter


def grid_search(main, args, max_epoch=10, gpu_available=0.5, num_samples=10):
    apply_search_wrapper()
    main = change_main(main)
    config, scheduler, reporter = get_utils(args, max_epoch)

    result = tune.run(
        main,
        resources_per_trial={"cpu": 4, "gpu": gpu_available},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    result.get_best_trial("loss", "min", "last")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

