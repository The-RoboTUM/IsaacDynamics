# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def delta_entropy(kde_model_obs, kde_model_actions, x_min, x_max, u_min, u_max, num_points=1000):
    # Evaluation grid obs
    y_grid = np.linspace(x_min, x_max, num_points)
    dy = (x_max - x_min) / num_points

    # Evaluation grid actions
    u_grid = np.linspace(u_min, u_max, num_points)
    du = (u_max - u_min) / num_points

    # Evaluate KDE at grid points
    p_y = kde_model_obs.evaluate(y_grid)
    p_y = np.clip(p_y, 1e-12, None)
    p_u = kde_model_actions.evaluate(u_grid)
    p_u = np.clip(p_u, 1e-12, None)

    # Uniform prior density
    uniform_density_obs = 1.0 / (x_max - x_min)
    uniform_density_actions = 1.0 / (u_max - u_min)

    # Compute pointwise KL terms
    kl_terms_obs = p_y * np.log2(p_y / uniform_density_obs)
    kl_terms_actions = p_u * np.log2(p_u / uniform_density_actions)

    # Approximate the integral with the trapezoidal rule
    kl_div_obs = np.sum(kl_terms_obs) * dy
    kl_div_actions = np.sum(kl_terms_actions) * du

    # Control effort is negative KL divergence
    delta_H = -kl_div_obs - kl_div_actions
    return delta_H, -kl_div_obs, -kl_div_actions
