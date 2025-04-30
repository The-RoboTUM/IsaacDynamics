# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def control_effort(kde_model, obs_values, delta_t, x_min, x_max):
    # Uniform density
    uniform_density = 1.0 / (x_max - x_min)

    # Evaluate KDE at each value
    p_values = kde_model.evaluate(obs_values)

    # Numerical stability
    p_values = np.clip(p_values, 1e-12, None)

    # KL terms at each step
    kl_terms = np.log2(p_values / uniform_density)

    # Sum over all steps and multiply by delta_time
    cont_effort = -(np.sum(p_values * kl_terms) * delta_t)

    # TODO: Change the sign to symbolize the change of entropy
    cont_effort *= -1

    return cont_effort
