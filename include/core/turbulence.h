// SPDX-FileCopyrightText: Copyright (c) 2019, 2021-2025 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_turbulenceModel_h
#define lethe_turbulenceModel_h

enum turbulenceModelID : unsigned int
{
  none   = 0,
  kOmega = 1
};

/**
 * @brief Solution fields of the different physics that are used as an indicator
 * for multiple purposes (e.g. adaptive mesh refinement, solid domain
 * constraints).
 */
enum class Variable : unsigned int
{ /// Velocity vector field from fluid dynamics
  velocity = 0,
  /// Pressure scalar field from fluid dynamics
  pressure = 1,
  /// turbulence kinetic energy
  turbK = 2,
  /// turbulence eddy frequency
  turbOmega = 3,
  /// eddy viscosity
  turbNut = 4
};

/**
 * @brief Utility function used for parsing physics-based
 * parameters
 *
 */
inline turbulenceModelID
get_turbulenceModel_id(std::string turbulenceModel_name)
{
  if (turbulenceModel_name == "none")
    return turbulenceModelID::none;
  else if (turbulenceModel_name == "kOmega")
    return turbulenceModelID::kOmega;
  else
    AssertThrow(false,
                dealii::StandardExceptions::ExcMessage(
                  "An unknown turbulenceModel name was requested"));
}

#endif
