// SPDX-FileCopyrightText: Copyright (c) 2023-2025 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_idealgas_compressible_navier_stokes_assembler_h
#define lethe_idealgas_compressible_navier_stokes_assembler_h

#include <core/simulation_control.h>

#include <solvers/copy_data.h>
#include <solvers/navier_stokes_assemblers.h>
#include <solvers/navier_stokes_scratch_data.h>


/**
 * @brief Class that assembles the core of the ideal gas compressible Navier-Stokes equations.
 *  According to the following weak form:
 * \f$\nabla \cdot (\rho \mathbf{u}) + \rho \mathbf{u} \cdot \nabla \mathbf{u} -
 * \nabla p - \mu \nabla^2 \mathbf{u} = 0\f$ with a full GLS stabilization
 * including the laplacian of the test function.
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */

template <int dim>
class GLSIdealGasCompressibleNavierStokesAssemblerCore
  : public NavierStokesAssemblerBase<dim>
{
public:
  GLSIdealGasCompressibleNavierStokesAssemblerCore(
    const std::shared_ptr<SimulationControl> &simulation_control)
    : simulation_control(simulation_control)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(const NavierStokesScratchData<dim>   &scratch_data,
                  StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(const NavierStokesScratchData<dim>   &scratch_data,
               StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  const std::shared_ptr<SimulationControl> simulation_control;
};


/**
 * @brief Class that assembles the transient time arising from BDF time
 * integration for the idealgas compressible Navier Stokes equations. For
 * example, if a BDF1 scheme is chosen, the following is assembled
 * \f$\frac{(\rho \mathbf{u})^{t+\Delta t}-(\rho \mathbf{u})^{t}}{\Delta t} +
 * \frac{(\psi \mathbf{p})^{t+\Delta t}-(\psi \mathbf{p})^{t}}{\Delta t} \f$
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */
template <int dim>
class GLSIdealGasCompressibleNavierStokesAssemblerBDF
  : public NavierStokesAssemblerBase<dim>
{
public:
  GLSIdealGasCompressibleNavierStokesAssemblerBDF(
    const std::shared_ptr<SimulationControl> &simulation_control)
    : simulation_control(simulation_control)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(const NavierStokesScratchData<dim>   &scratch_data,
                  StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(const NavierStokesScratchData<dim>   &scratch_data,
               StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  const std::shared_ptr<SimulationControl> simulation_control;
};

#endif
