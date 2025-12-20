// SPDX-FileCopyrightText: Copyright (c) 2021-2025 The Lethe Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later

/*
 * This class provides an interface for multiphysics simulations by enabling
 * the solution of multiple auxiliary physics on top of a computational
 * fluid dynamics simulation. The auxiliary physics are stored in a map
 * whose keys are the Parameters::PhysicsID int enum.
 */

#ifndef lethe_turbulence_interface_h
#define lethe_turbulence_interface_h

#include <core/exceptions.h>
#include <core/turbulence.h>
#include <core/parameters_turbulence.h>
#include <core/simulation_control.h>
#include <core/solid_base.h>
#include <core/vector.h>

#include <solvers/auxiliary_turbulence_model.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <map>
#include <memory>

using namespace dealii;

template <int dim>
class TurbulenceInterface
{
public:
  /** @brief Construct the Multiphysics interface from the simulation parameters.
   * Depending on which multiphysics element is enabled, the appropriate
   * auxiliary physics is instantiated.
   *
   */
  TurbulenceInterface(
    const SimulationParameters<dim> &nsparam,
    std::shared_ptr<parallel::DistributedTriangulationBase<dim>>
                                       p_triangulation,
    std::shared_ptr<SimulationControl> p_simulation_control,
    ConditionalOStream                &p_pcout);

  /**
   * @brief Default destructor.
   */
  virtual ~TurbulenceInterface() = default;

  std::vector<turbulenceModelID>
  get_active_turbulenceModels()
  {
    return active_turbulenceModels;
  }

  /**
   * @brief Write physic solved in the terminal
   *
   * @param turbulenceModel_id number associated with auxiliary turbulence model in turbulence.h
   */
  void
  announce_turbulenceModel(const turbulenceModelID turbulenceModel_id)
  {
    if (turbulenceModel_id == turbulenceModelID::kOmega)
      {
        announce_string(pcout, "kOmega");
      }
    else if (turbulenceModel_id == turbulenceModelID::none)
      {
        announce_string(pcout, "none");
      } 
  }

  /**
   * @brief Call for the solution of the physics that should be solved
   *
   * @param fluid_dynamics_has_been_solved Boolean that states if the fluid dynamics has been
   * already solved or not. See the map `solve_pre_fluid` to know which
   * subphysics are solved before the fluid dynamics.
   *
   * @param time_stepping_method Time-Stepping method with which the assembly is called
   */
  void
  solve(const bool fluid_dynamics_has_been_solved,
        const Parameters::SimulationControl::TimeSteppingMethod
          time_stepping_method)
  {
    // Loop through all the elements in the turbulence equation map. Consequently, iturb is
    // an std::pair where iturb.first is the turbulenceModelID and iturb.second is the
    // AuxiliaryPhysics pointer. This is how the map can be traversed
    // sequentially.
    for (auto &iturb : turbulenceModels)
      {
        // If iturb.first should be solved BEFORE fluid dynamics
        if (!fluid_dynamics_has_been_solved && solve_pre_fluid[iturb.first])
          solve_equations(iturb.first, time_stepping_method);

        // If iphys.first should be solved AFTER fluid dynamics OR if is not
        // present in solve_pre_fluid map
        else if (fluid_dynamics_has_been_solved &&
                 (!solve_pre_fluid[iturb.first] ||
                  solve_pre_fluid.count(iturb.first) == 0))
          solve_equations(iturb.first, time_stepping_method);
      }

    for (auto &iturb : block_turbulenceModels)
      {
        // If iphys.first should be solved BEFORE fluid dynamics
        if (!fluid_dynamics_has_been_solved && solve_pre_fluid[iturb.first])
          solve_block_equations(iturb.first, time_stepping_method);

        // If iphys.first should be solved AFTER fluid dynamics OR if is not
        // present in solve_pre_fluid map
        else if (fluid_dynamics_has_been_solved &&
                 (!solve_pre_fluid[iturb.first] ||
                  solve_pre_fluid.count(iturb.first) == 0))
          solve_block_equations(iturb.first, time_stepping_method);
      }
  }

  /**
   * @brief Call for the solution of a single physic
   *
   * @param time_stepping_method Time-Stepping method with which the assembly is called
   */
  void
  solve_equations(const turbulenceModelID turbulenceModel_id,
                const Parameters::SimulationControl::TimeSteppingMethod
                  time_stepping_method)
  {
    // Announce physic solved (verbosity = physics_solving_strategy.verbosity)
    if (verbosity.at(turbulenceModel_id) != Parameters::Verbosity::quiet)
      announce_turbulenceModel(turbulenceModel_id);

    AssertThrow(std::find(active_turbulenceModels.begin(),
                          active_turbulenceModels.end(),
                          turbulenceModel_id) != active_turbulenceModels.end(),
                ExcInternalError());

    turbulenceModels[turbulenceModel_id]->time_stepping_method = time_stepping_method;
    turbulenceModels[turbulenceModel_id]->solve_governing_system();
    turbulenceModels[turbulenceModel_id]->modify_solution();
  }

  /**
   * @brief Call for the solution of a single block turbulence model
   *
   * @param time_stepping_method Time-Stepping method with which the assembly is called
   */
  void
  solve_block_equations(const turbulenceModelID turbulenceModel_id,
                      const Parameters::SimulationControl::TimeSteppingMethod
                        time_stepping_method)
  {
    // Announce physic solved (verbosity = physics_solving_strategy.verbosity)
    if (verbosity.at(turbulenceModel_id) != Parameters::Verbosity::quiet)
      announce_turbulenceModel(turbulenceModel_id);

    AssertThrow(std::find(active_turbulenceModels.begin(),
                          active_turbulenceModels.end(),
                          turbulenceModel_id) != active_turbulenceModels.end(),
                ExcInternalError());

    block_turbulenceModels[turbulenceModel_id]->time_stepping_method = time_stepping_method;
    block_turbulenceModels[turbulenceModel_id]->solve_governing_system();
    block_turbulenceModels[turbulenceModel_id]->modify_solution();
  }

  /**
   * @brief Gather and return vector of output structs that are particular to some applications.
   *
   * @return Vector of OutputStructs that will be used to write the output results as VTU files. This is a variant for GlobalVectorType.
   */
  std::vector<OutputStruct<dim, GlobalVectorType>>
  gather_output_hook_global_vector()
  {
    std::vector<OutputStruct<dim, GlobalVectorType>> solution_output_structs;
    for (auto &iturb : turbulenceModels)
      {
        std::vector<OutputStruct<dim, GlobalVectorType>> output_structs =
          iturb.second->gather_output_hook();
        for (auto &output_struct : output_structs)
          solution_output_structs.push_back(output_struct);
      }

    return solution_output_structs;
  }

  /**
   * @brief Gather and return vector of output structs that are particular to some applications.
   *
   * @return Vector of OutputStructs that will be used to write the output results as VTU files. This is a variant for GlobalBlockVectorType.
   */
  std::vector<OutputStruct<dim, GlobalBlockVectorType>>
  gather_output_hook_global_block_vector()
  {
    std::vector<OutputStruct<dim, GlobalBlockVectorType>>
      solution_output_structs;
    for (auto &iturb : block_turbulenceModels)
      {
        std::vector<OutputStruct<dim, GlobalBlockVectorType>> output_structs =
          iturb.second->gather_output_hook();
        for (auto &output_struct : output_structs)
          solution_output_structs.push_back(output_struct);
      }

    return solution_output_structs;
  }

  /**
   * @brief Carry out the operations required to finish a simulation correctly for
   * all auxiliary physics.
   */
  void
  finish_simulation()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->finish_simulation();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->finish_simulation();
      }
  }

  /**
   * @brief Rearrange vector solution correctly for transient simulations for
   * all auxiliary physics.
   *
   * @param fluid_dynamics_has_been_solved Boolean that states if the fluid dynamics has been
   * already solved or not. See the map `solve_pre_fluid` to know which
   * subphysics are solved before the fluid dynamics.
   */
  void
  percolate_time_vectors(const bool fluid_dynamics_has_been_solved)
  {
    // Loop through all the elements in the turbulenceModels map. Consequently, iturb is
    // an std::pair where iturb.first is the turbulenceModelID and iturb.second is the
    // AuxiliaryPhysics pointer. This is how the map can be traversed
    // sequentially.
    for (auto &iturb : turbulenceModels)
      {
        // If iturb.first should be percolated BEFORE fluid dynamics is solved
        if (!fluid_dynamics_has_been_solved && solve_pre_fluid[iturb.first])
          iturb.second->percolate_time_vectors();

        // If iturb.first should be percolated AFTER fluid dynamics is solved OR
        // if is not present in solve_pre_fluid map
        else if (fluid_dynamics_has_been_solved &&
                 (!solve_pre_fluid[iphys.first] ||
                  solve_pre_fluid.count(iphys.first) == 0))
          iturb.second->percolate_time_vectors();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        // If iturb.first should be percolated BEFORE fluid dynamics is solved
        if (!fluid_dynamics_has_been_solved && solve_pre_fluid[iturb.first])
          iturb.second->percolate_time_vectors();

        // If iphys.first should be percolated AFTER fluid dynamics is solved OR
        // if is not present in solve_pre_fluid map
        else if (fluid_dynamics_has_been_solved &&
                 (!solve_pre_fluid[iturb.first] ||
                  solve_pre_fluid.count(iturb.first) == 0))
          iturb.second->percolate_time_vectors();
      }
  }

  /**
   * @param Update the boundary conditions of the auxiliary physics if they are time-dependent
   */
  void
  update_boundary_conditions()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->update_boundary_conditions();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->update_boundary_conditions();
      }
  }

  /**
   * @brief Postprocess the auxiliary physics results. Post-processing this case implies
   * the calculation of all derived quantities using the solution vector
   * of the physics. It does not concern the output of the solution using
   * the DataOutObject, which is accomplished through the
   * attach_solution_to_output function
   */
  void
  postprocess(bool first_iteration)
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->postprocess(first_iteration);
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->postprocess(first_iteration);
      }
  }


  /**
   * @brief Prepare the auxiliary physics for mesh adaptation
   */
  void
  prepare_for_mesh_adaptation()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->pre_mesh_adaptation();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->pre_mesh_adaptation();
      }
  }

  /**
   * @brief Interpolate solution onto new mesh
   */
  void
  post_mesh_adaptation()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->post_mesh_adaptation();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->post_mesh_adaptation();
      }
  }



  /**
   * @brief Set-up the DofHandler and the degree of freedom associated with the physics.
   */
  void
  setup_dofs()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->setup_dofs();
      }
    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->setup_dofs();
      }
  };



  /**
   * @brief Sets-up the initial conditions associated with the physics. Generally, physics
   * only support imposing nodal values, but some physics additionally
   * support the use of L2 projection or steady-state solutions.
   */
  void
  set_initial_conditions()
  {
    for (auto &iturb : turbulenceModels)
      {
        iturb.second->set_initial_conditions();
      }

    for (auto &iturb : block_turbulenceModels)
      {
        iturb.second->set_initial_conditions();
      }
  };

  /**
   * @brief Call for the solution of the linear system of an auxiliary physics.
   *
   */
  void
  solve_linear_system(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());

    turbulenceModels[turbulenceModel_id]->solve_linear_system();
  };

  /**
   * @brief fluid_dynamics_is_block Verify if the fluid dynamics solution
   * is stored as a block vector or not.
   *
   * @return boolean value indicating the fluid dynamics is stored as a block vector
   */

  bool
  fluid_dynamics_is_block() const
  {
    return block_turbulence_solutions.find(TurbulenceModelID::fluid_dynamics) !=
           block_turbulence_solutions.end();
  }



  /**
   * @brief Request a DOF handler for a given physics ID
   *
   * @param turbulenceModel_id The turbulence model of the DOF handler being requested
   *
   * @return Reference to the DOF handler of the requested physics
   */
  const DoFHandler<dim> &
  get_dof_handler(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());

    return *turbulenceModels[turbulenceModel_id]->get_dof_handler();
  }

  /**
   * @brief Request the reference to the present solution of a given physics
   *
   * @param turbulenceModel_id The turbulence model ID of the solution being requested
   *
   * @return Reference to the solution vector of the requested physics
   */
  const GlobalVectorType &
  get_solution(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *turbulenceModels[turbulenceModel_id]->get_solution();
  }

  /**
   * @brief Request the reference to the present filtered solution of a given
   * physics (used in VOF or CahnHilliard physics for STF calculation in the
   * momentum balance)
   *
   * @param[in] turbulenceModel_id ID of the turbulence model for which the filtered solution is
   * being requested
   *
   * @return Reference to the requested filtered solution vector.
   */
  const GlobalVectorType &
  get_filtered_solution(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *turbulenceModels[turbulenceModel_id]->get_filtered_solution();
  }

  /**
   * @brief Request the reference to the present block solution of a given
   * physics
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the solution being requested
   */
  const GlobalBlockVectorType &
  get_block_solution(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *block_turbulence_solutions[turbulenceModel_id];
  }


  /**
   * @brief Request the reference to the time-average solution of a given physics
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the solution being requested
   */
  const GlobalVectorType &
  get_time_average_solution(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *turbulenceModels[turbulenceModel_id]->get_time_average_solution();
  }

  /**
   * @brief Request the reference to the present block average solution of a
   * given physics
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the solution being requested
   */
  const GlobalBlockVectorType &
  get_block_time_average_solution(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *block_turbulence_solutions[turbulenceModel_id];
  }

  /**
   * @brief Request the solid objects. Used an auxiliary physics
   * needs to apply a boundary condition on a solid through
   * Nitsche immersed boundary method.
   *
   * @param[in] number_solids The number of solids declared in the parameter
   * file. The value is used to ensure that at least one solid has been
   * declared.
   *
   * @note The method is called only in
   * HeatTransfer<dim>::assemble_nitsche_heat_restriction,
   * which is itself called only if number_solids > 0
   */
  const std::vector<std::shared_ptr<SolidBase<dim, dim>>> &
  get_solids([[maybe_unused]] const int number_solids)
  {
    Assert(number_solids > 0, NoSolidWarning("the"));
    AssertThrow(solids != nullptr,
                dealii::ExcMessage("solids is not initialized"));
    return *solids;
  }

  /**
   * @brief Request the reference to the present solution vector of the
   * projected phase fraction gradient (PFG)
   */
  const GlobalVectorType &
  get_projected_phase_fraction_gradient_solution();

  /**
   * @brief Request the reference to the present solution of the curvature
   */
  const GlobalVectorType &
  get_curvature_solution();

  /**
   * @brief Request the reference to the projected curvature DOF handler
   */
  const DoFHandler<dim> &
  get_curvature_dof_handler();

  /**
   * @brief Request the reference to the projected phase fraction gradient (PFG)
   * DOF handler
   */
  const DoFHandler<dim> &
  get_projected_phase_fraction_gradient_dof_handler();

  /**
   * @brief Request shared pointer to immersed solid shape
   */
  std::shared_ptr<Shape<dim>>
  get_immersed_solid_shape();

  /**
   * @brief Share immersed solid shape
   *
   * @param[in] shape The reference to the shared pointer pointing to the
   * immersed solid shape
   */
  void
  set_immersed_solid_shape(const std::shared_ptr<Shape<dim>> &shape);

  /**
   * @brief Request the reference to the vector of previous solutions of a given
   * physics
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the solution being requested
   */
  const std::vector<GlobalVectorType> &
  get_previous_solutions(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *turbulenceModels[turbulenceModel_id]->get_previous_solutions();
  }


  /**
   * @brief Request the reference to the vector of previous solutions of a given
   * block physics
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the solution being requested
   */
  const std::vector<GlobalBlockVectorType> &
  get_block_previous_solutions(const turbulenceModelID turbulenceModel_id)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    return *block_turbulence_solutions[turbulenceModel_id];
  }

  /**
   * @brief Sets the shared pointer to the DOFHandler of the physics in the
   * multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model of the DOF handler being requested
   *
   * @param[in] dof_handler Shared pointer to the dof handler for which the
   * reference is stored
   */
  void
  set_dof_handler(const turbulenceModelID                  turbulenceModel_id,
                  std::shared_ptr<DoFHandler<dim>> dof_handler)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    turbulenceModel_dof_handler[turbulenceModel_id] = dof_handler;
  }

  /**
   * @brief Sets the shared pointer to the vector of the SolidBase object. This
   * allows the use of the solid base object in multiple physics at the same
   * time.
   *
   * @param[in] solids_input Shared pointer to the vector of solidBase object
   */
  void
  set_solid(std::shared_ptr<std::vector<std::shared_ptr<SolidBase<dim, dim>>>>
              solids_input)
  {
    solids = solids_input;
  }

  /**
   * @brief Sets the shared pointer to the solution of the physics in the
   * multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model ID the present solution being set
   *
   * @param[in] solution_vector Shared pointer to the solution vector of the
   * physics
   */
  void
  set_solution(const turbulenceModelID                   turbulenceModel_id,
               std::shared_ptr<GlobalVectorType> solution_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    turbulenceModel_solutions[turbulenceModel_id] = solution_vector;
  }

  /**
   * @brief Sets the shared pointer to the filtered solution of the physics in
   * the multiphysics interface (used in VOF or CahnHilliard physics for STF
   * calculation in the momentum balance)
   *
   * @param[in] turbulenceModel_id ID of the turbulence model for which the filtered solution is
   * being set
   *
   * @param[in] filtered_solution_vector Shared pointer to the filtered solution
   * vector of the physics; this was implemented for VOF and CahnHilliard
   * physics
   */
  void
  set_filtered_solution(
    const turbulenceModelID                   turbulenceModel_id,
    std::shared_ptr<GlobalVectorType> filtered_solution_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    turbulenceModel_filtered_solutions[turbulenceModel_id] = filtered_solution_vector;
  }


  /**
   * @brief Sets the shared pointer to the time-average solution of the turbulence model in the multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the time averaged solution being
   * set
   *
   * @param[in] solution_vector The shared pointer to the time averaged solution
   * vector of the physics
   */
  void
  set_time_average_solution(const turbulenceModelID                   turbulenceModel_id,
                            std::shared_ptr<GlobalVectorType> solution_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    turbulenceModel_time_average_solutions[turbulenceModel_id] = solution_vector;
  }

  /**
   * @brief Sets the reference to the solution of the physics in the multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the DOF handler being requested
   *
   * @param[in] solution_vector The shared pointer to the solution vector of the
   * requested physics
   */
  void
  set_block_solution(const turbulenceModelID                        turbulenceModel_id,
                     std::shared_ptr<GlobalBlockVectorType> solution_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    block_turbulenceModel_solutions[turbulenceModel_id] = solution_vector;
  }

  /**
   * @brief Sets the shared pointer to the time-average solution of the block
   * physics in the multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model ID of the time averaged block vector
   * solution being set
   *
   * @param[in] solution_vector The shared pointer to the block solution vector
   * of the physics
   */
  void
  set_block_time_average_solution(
    const turbulenceModelID                        turbulenceModel_id,
    std::shared_ptr<GlobalBlockVectorType> solution_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    block_turbulenceModel_time_average_solutions[turbulenceModel_id] = solution_vector;
  }

  /**
   * @brief Sets the pointer to the vector of previous solutions of the physics in the multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model of the DOF handler
   *
   * @param[in] previous_solutions_vector The shared pointer to the vector of
   * previous solutions
   */
  void
  set_previous_solutions(
    const turbulenceModelID                                turbulenceModel_id,
    std::shared_ptr<std::vector<GlobalVectorType>> previous_solutions_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    turbulenceModel_previous_solutions[turbulenceModel_id] = previous_solutions_vector;
  }

  /**
   * @brief Sets the pointer to the vector of previous solutions of the block
   * physics in the multiphysics interface
   *
   * @param[in] turbulenceModel_id The turbulence model of the DOF handler
   *
   * @param[in] previous_solutions_vector The shared pointer to the vector of
   * previous block vector solutions
   */
  void
  set_block_previous_solutions(
    const turbulenceModelID                                turbulenceModel_id,
    std::shared_ptr<std::vector<GlobalBlockVectorType>>
      previous_solutions_vector)
  {
    AssertThrow((std::find(active_turbulenceModels.begin(),
                           active_turbulenceModels.end(),
                           turbulenceModel_id) != active_turbulenceModels.end()),
                ExcInternalError());
    block_turbulenceModel_previous_solutions[turbulenceModel_id] = previous_solutions_vector;
  }

  /**
   * @brief Mesh refinement according to an auxiliary physic parameter
   *
   * @param ivar The current element of the map simulation_parameters.mesh_adaptation.variables
   *
   * @param estimated_error_per_cell The deal.II vector of estimated_error_per_cell
   */
  virtual void
  compute_kelly(const std::pair<const Variable,
                                Parameters::MultipleAdaptationParameters> &ivar,
                dealii::Vector<float> &estimated_error_per_cell)
  {
    for (auto &iturbulenceModel : turbulenceModels)
      {
        iturbulenceModel.second->compute_kelly(ivar, estimated_error_per_cell);
      }
    for (auto &iturbulenceModel : block_turbulenceModels)
      {
        iturbulenceModel.second->compute_kelly(ivar, estimated_error_per_cell);
      }
  };

  /**
   * @brief Prepares auxiliary physics to write simulation checkpoint
   */
  virtual void
  write_checkpoint()
  {
    for (auto &iturbulenceModel : turbulenceModels)
      {
        iturbulenceModel.second->write_checkpoint();
      }
    for (auto &iturbulenceModel : block_turbulenceModels)
      {
        iturbulenceModel.second->write_checkpoint();
      }
  };

  /**
   * @brief Read solution from checkpoint from auxiliary physics
   *
   */
  virtual void
  read_checkpoint()
  {
    for (auto &iturbulenceModel : turbulenceModels)
      {
        iturbulenceModel.second->read_checkpoint();
      }
    for (auto &iturbulenceModel : block_turbulenceModels)
      {
        iturbulenceModel.second->read_checkpoint();
      }
  };

private:
  const Parameters::Multiphysics             turbulence_parameters;
  std::map<TurbulenceModelID, Parameters::Verbosity> verbosity;
  ConditionalOStream                         pcout;

  // Data structure to store all physics which were enabled
  std::vector<TurbulenceModelID> active_turbulenceModels;


  // Map that states if the physics are solved before the fluid dynamics
  std::map<turbulenceModelID, bool> solve_pre_fluid{{none, false},
                                                    {kOmega, false}};

  // Auxiliary physics are stored within a map of shared pointer to ensure
  // proper memory management.
  std::map<TurbulenceModelID, std::shared_ptr<AuxiliaryTurbulenceModel<dim, GlobalVectorType>>>
    turbulenceModels;

  std::map<turbulenceModelID,
           std::shared_ptr<AuxiliaryTurbulenceModel<dim, GlobalBlockVectorType>>>
    block_turbulenceModels;

  /// Map of physics and shared pointers to their respective DoFHandler
  std::map<TurbulenceModelID, std::shared_ptr<DoFHandler<dim>>> turbulenceModel_dof_handler;

  /// Shared pointer to the vector containing shared pointers to solid objects
  std::shared_ptr<std::vector<std::shared_ptr<SolidBase<dim, dim>>>> solids;

  /// Map of physics and shared pointers to their respective solutions.
  std::map<TurbulenceModelID, std::shared_ptr<std::vector<GlobalVectorType>>> turbulenceModel_solutions;

  /**
   * Map of physics and shared pointers to their respective solutions.
   * Same as MultiphysicsInterface::physics_solutions, but used with
   * BlockVector.
   */
  std::map<TurbulenceModelID, std::shared_ptr<std::vector<GlobalVectorType>>>
    block_turbulenceModel_solutions;

  /**
   * Map of physics and shared pointers to their respective filtered solutions.
   * These solutions are used with both VOF and Cahn-Hilliard multiphase flow
   * approaches for surface tension force calculation in the momentum equation.
   */
  std::map<TurbulenceModelID, std::shared_ptr<std::vector<GlobalVectorType>>>
    turbulenceModel_filtered_solutions;

  /**
   * Map of physics and shared pointers to their respective filtered solutions.
   * Same as MultiphysicsInterface::physics_filtered_solutions, but used with
   * BlockVector.
   */
  std::map<TurbulenceModelID, std::shared_ptr<std::vector<GlobalVectorType>>>
    block_turbulenceModel_filtered_solutions;

  /**
   * Map of physics and shared pointers to their respective vector of previous
   * solutions.
   */
  std::map<PhysicsID, std::shared_ptr<std::vector<GlobalVectorType>>>
    physics_previous_solutions;

  /**
   * Map of physics and shared pointers to their respective vector of previous
   * solutions.
   * Same as MultiphysicsInterface::physics_previous_solutions, but used with
   * BlockVector.
   */
  std::map<TurbulenceModelID, std::shared_ptr<std::vector<GlobalVectorType>>>
    block_turbulenceModel_previous_solutions;

  /**
   * Map of physics and shared pointers to their respective time-averaged
   * solutions.
   */
  std::map<TurbulenceModelID, std::shared_ptr<GlobalVectorType>>
    turbulenceModel_time_average_solutions;

  /**
   * Map of physics and shared pointers to their respective time-averaged
   * solutions.
   * Same as MultiphysicsInterface::physics_time_average_solutions, but used
   * with BlockVector.
   */
  std::map<TurbulenceModelID, std::shared_ptr<GlobalBlockVectorType>>
    block_turbulenceModel_time_average_solutions;

  /// Shared pointer to immersed solid shapes to be used by auxiliary physics
  std::shared_ptr<Shape<dim>> immersed_solid_shape;

  /**
   * Map of physics and shared pointers to their respective previous (n-1)
   * solutions.
   */
  std::map<TurbulenceModelID, std::shared_ptr<GlobalVectorType>> turbulenceModel_solutions_m1;
  std::map<TurbulenceModelID, std::shared_ptr<GlobalBlockVectorType>>
    block_turbulenceModel_solutions_m1;

  // Checks the required dependencies between multiphase models and handles the
  // corresponding assertions
  void
  inspect_multiphysics_models_dependencies(
    const SimulationParameters<dim> &nsparam);
};


#endif
