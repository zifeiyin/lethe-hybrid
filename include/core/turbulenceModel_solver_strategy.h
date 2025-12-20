// SPDX-FileCopyrightText: Copyright (c) 2019-2025 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

#ifndef lethe_turbulenceModel_solver_strategy_h
#define lethe_turbulenceModel_solver_strategy_h

#include <core/parameters.h>

template <typename VectorType>
class TurbulenceModelSolver;

/**
 * @brief Base class that works as an interface for all solver strategies (either non-linear or linear) for all systems of equations.
 *
 */
template <typename VectorType>
class TurbulenceModelSolverStrategy
{
public:
  /**
   * @brief Constructor for the non-linear solver strategies.
   *
   * @param[in] turbulenceModel_solver A pointer to the turbulenceModel solver to which the
   * solving strategy is attached.
   *
   * @param[in] param Non-linear solver parameters as specified in the
   * simulation parameter file.
   *
   */
  TurbulenceModelSolverStrategy(TurbulenceModelSolver<VectorType>         *turbulenceModel_solver,
                        const Parameters::NonLinearSolver &params);

  /**
   * @brief Constructor for the linear solver strategy.
   *
   * @param[in] turbulenceModel_solver A pointer to the turbulenceModel solver to which the
   * solving strategy is attached.
   *
   */
  TurbulenceModelSolverStrategy(TurbulenceModelSolver<VectorType> *turbulenceModel_solver);

  /**
   * @brief Destructor.
   *
   */
  virtual ~TurbulenceModelSolverStrategy()
  {}

  /**
   * @brief Solve the system of equations.
   *
   */
  virtual void
  solve() = 0;


  /**
   * @brief Get the current newton iteration.
   *
   * @return Iteration number.
   *
   */
  inline unsigned int
  get_current_newton_iteration() const
  {
    return outer_iteration;
  }

protected:
  /**
   * @brief Physics solver for which we need a non-linear solver.
   *
   */
  TurbulenceModelSolver<VectorType> *turbulenceModel_solver;

  /**
   * @brief Non linear solver parameters.
   *
   */
  Parameters::NonLinearSolver params;

  /**
   * @brief Number of current Newton iteration.
   *
   */
  unsigned int outer_iteration;
};

template <typename VectorType>
TurbulenceModelSolverStrategy<VectorType>::TurbulenceModelSolverStrategy(
  TurbulenceModelSolver<VectorType>         *turbulenceModel_solver,
  const Parameters::NonLinearSolver &params)
  : turbulenceModel_solver(turbulenceModel_solver)
  , params(params)
  , outer_iteration(0)
{}

template <typename VectorType>
TurbulenceModelSolverStrategy<VectorType>::TurbulenceModelSolverStrategy(
  TurbulenceModelSolver<VectorType> *turbulenceModel_solver)
  : turbulenceModel_solver(turbulenceModel_solver)
{}

#endif
