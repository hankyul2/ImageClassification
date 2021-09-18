from ray import tune


def objective(x, a, b):
    return a * (x ** 0.5) + b

def trainable(config):
    # config (dict): A dict of hyperparameters.

    for x in range(20):
        score = objective(x, config["a"], config["b"])

        tune.report(score=score)  # This sends the score to Tune.

space = {
    "a": tune.uniform(0, 1),
    "b": tune.uniform(0, 1),
}
tune.run(trainable, config=space, num_samples=10)