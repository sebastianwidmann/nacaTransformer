import optax
import numpy as np


def load_learning_rate_scheduler(config, name, total_steps):
    if name == 'constant':
        return config.learning_rate_end_value
    elif name == "warmup_cosine_decay":
        return optax.warmup_cosine_decay_schedule(
            init_value=1e-4, peak_value=3e-4,
            warmup_steps=int(0.2 * total_steps),
            decay_steps=total_steps, end_value=2.5e-5,
        )
    elif name == "sgdr":
        decay_steps = int(total_steps / config.sgdr_restarts)
        warmup_steps = int(config.warmup_fraction * decay_steps)

        # array structure of values = [init_value, peak_value, end_value]
        values = np.zeros((config.sgdr_restarts, 3))
        lr = config.learning_rate_end_value
        values[:, 0] = 2 * lr * (2 ** np.arange(0, config.sgdr_restarts, 1))
        values[:, 1] = 2 * values[:, 0]
        values[:, 2] = 0.5 * values[:, 0]
        values = np.flip(values, axis=0)

        lr_schedule = []
        for i in range(config.sgdr_restarts):
            lr_schedule.append(
                {"init_value": values[i, 0], "peak_value": values[i, 1],
                 "decay_steps": decay_steps, "warmup_steps": warmup_steps,
                 "end_value": values[i, 2]}
            )

        return optax.sgdr_schedule(lr_schedule)
