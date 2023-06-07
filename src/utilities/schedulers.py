import optax


def load_learning_rate_scheduler(name, total_steps):
    if name == 'constant':
        return 0.0003
    elif name == "warmup_cosine_decay":
        return optax.warmup_cosine_decay_schedule(
            init_value=1e-4, peak_value=3e-4,
            warmup_steps=int(0.2 * total_steps),
            decay_steps=total_steps, end_value=1e-5,
        )
    elif name == "sgdr":
        return optax.sgdr_schedule(
            [
                {"init_value": 3e-4, "peak_value": 4e-4,
                 "decay_steps": 7100, "warmup_steps": 710, "end_value": 2e-4},
                {"init_value": 2e-4, "peak_value": 3e-4,
                 "decay_steps": 7100, "warmup_steps": 710, "end_value": 1e-4},
                {"init_value": 1e-4, "peak_value": 2e-4,
                 "decay_steps": 7100, "warmup_steps": 710, "end_value":
                     5e-5},
                {"init_value": 5e-5, "peak_value": 1e-4,
                 "decay_steps": 7100, "warmup_steps": 710, "end_value":
                     2.5e-5},
            ]
        )
