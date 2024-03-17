import jax
import jax.numpy as jnp


# TODO test if this works as expected
def set_geometry_internal_value(batch, old_value, new_value):
    encoder_input = batch['encoder']
    decoder_input = batch['decoder']

    new_encoder_input = jnp.where(encoder_input[:, :, :, :] == old_value, new_value, encoder_input[:, :, :, :])
    new_decoder_input = jnp.where(decoder_input[:, :, :, :] == old_value, new_value, decoder_input[:, :, :, :])

    batch['encoder'] = new_decoder_input
    batch['decoder'] = new_decoder_input

    return batch


'''
https://en.wikipedia.org/wiki/Pressure_coefficient
'''


@jax.jit
def pressure_coefficient(decoder_input, mach, geometry_internal_value):
    # TODO try using inf or NAN for the inside of the geometry instead of
    # # Define thermodynamic properties of air at ICAO standard atmosphere
    T0 = 288.15  # [K] Total temperature
    p0 = 101325  # [Pa] Total pressure
    gamma = 1.4  # [-] Ratio of specific heats
    R = 287.058  # [J/(kg*K)] Specific gas constant for dry air
    rho0 = 1.225  # [kg/m^3] air density look at ICA0 standard atmosphere

    # # Normalise pressure by freestream pressure

    T = T0 / (1 + 0.5 * (gamma - 1) * mach ** 2)
    p_inf = p0 * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (-gamma / (gamma - 1))
    u_inf = mach * jnp.sqrt(gamma * R * T)

    # since the TfRecord files are normalised by p_inf themselves we do (p-1)/(0.5*rho0*p_inf*u_inf^2)
    denominator = (rho0 * u_inf ** 2) / (2 * p_inf)
    result = jnp.where(decoder_input[:, :, 0] != geometry_internal_value, (decoder_input[:, :, 0] - 1) / denominator,
                       geometry_internal_value)
    decoder_input = decoder_input.at[:, :, 0].set(result)
    return decoder_input


@jax.jit
def standardize_pressure_and_velocity(decoder_input, geometry_internal_value):
    h, w, c = decoder_input.shape

    for i in range(c):
        field_copy = jnp.copy(decoder_input[:, :, i])

        field_copy = jnp.where(field_copy == geometry_internal_value, jnp.nan, field_copy)

        mean = jnp.nanmean(field_copy)
        std_deviation = jnp.nanstd(field_copy)

        result = jnp.where(decoder_input[:, :, i] != geometry_internal_value,
                           (decoder_input[:, :, i] - mean) / std_deviation, geometry_internal_value)
        decoder_input = decoder_input.at[:, :, i].set(result)

    return decoder_input


def get_mach(label):
    mach = []
    label = label.tolist()

    for entry in label:
        entry = entry.decode("utf-8")
        entry = entry[2:-2]
        label_data = entry.split('_')
        mach.append(float(label_data[-1]))

    return mach


def get_mean_std(decoder_input, geometry_internal_value):
    h, w, c = decoder_input.shape

    mean = []
    std = []

    for i in range(c):
        pressure_field_copy = jnp.copy(decoder_input[:, :, i])
        pressure_field_copy = jnp.where(pressure_field_copy == geometry_internal_value, jnp.nan, pressure_field_copy)
        mean_i = jnp.nanmean(pressure_field_copy)
        mean.append(mean_i)
        std_i = jnp.nanstd(pressure_field_copy)
        std.append(std_i)

    return (mean, std)


vectorized_get_mean_std = jax.vmap(get_mean_std, in_axes=(0, None))

@jax.jit
def reverse_standardize(fields, mean_std, geometry_internal_value):
    h, w, c = fields.shape
    mean, std = mean_std

    for i in range(c):
        result = jnp.where(fields[:, :, i] != geometry_internal_value, fields[:, :, i] * std[i] + mean[i],
                           geometry_internal_value)
        fields = fields.at[:, :, i].set(result)

    return fields

@jax.jit
def reverse_standardize_batched(y_test, predictions, mean_std, geometry_internal_value):
    vectorized_reverse_standardize = jax.vmap(reverse_standardize, in_axes=(0, 0, None))

    y_test = vectorized_reverse_standardize(y_test, mean_std, geometry_internal_value)
    predictions = vectorized_reverse_standardize(predictions, mean_std, geometry_internal_value)

    return y_test, predictions


def pressure_preprocessing(batch, config):
    type = config.pressure_preprocessing.type
    range = config.pressure_preprocessing.new_range
    geometry_internal_value = config.internal_geometry.value

    decoder_input = batch['decoder']
    label = batch['label']
    label_data = get_mach(label)

    mach = jnp.array(label_data)
    range_array = jnp.array(range)

    internal_value = geometry_internal_value

    vectorized_pressure_coefficient = jax.vmap(pressure_coefficient, in_axes=(0, 0, None))
    vectorized_standardize_all = jax.vmap(standardize_pressure_and_velocity, in_axes=(0, None))

    if type == 'standardize_all':
        batch['decoder'] = vectorized_standardize_all(decoder_input, internal_value)
    elif type == 'coefficient':
        batch['decoder'] = vectorized_pressure_coefficient(decoder_input, mach, internal_value)
    else:
        raise Exception("No proper normalization specified")

    return batch
