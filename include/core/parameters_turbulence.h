// SPDX-FileCopyrightText: Copyright (c) 2021-2025 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/*
 * This file defines the parameters in the parameter namespace
 * that pertain to multiphysics simulations
 */

#ifndef lethe_parameters_turbulence_h
#define lethe_parameters_turbulence_h

#include <core/parameters.h>

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace Parameters
{

  /**
   * @brief KOmega - Defines the parameters kOmega model.
   * Has to be declared before member creation in Turbulence structure.
   */
  struct KOmega
  {
    double comega1;

    void
    declare_parameters(ParameterHandler &prm) const;
    void
    parse_parameters(ParameterHandler &prm, const Dimensionality &dimensions);
  }


  /**
   * @brief Turbulence - the parameters for turbulence simulations
   * and handles sub-physics parameters.
   */
  struct Turbulence
  {
    bool none;
    bool kOmega;

    // subparameters for heat_transfer
    bool viscous_dissipation;
    bool buoyancy_force;

    Parameters::KOmega          kOmega_parameters;

    void
    declare_parameters(ParameterHandler &prm) const;
    void
    parse_parameters(ParameterHandler &prm, const Dimensionality &dimensions);
  };
} // namespace Parameters
#endif
