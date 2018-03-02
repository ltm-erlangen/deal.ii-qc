
// a source file which contains definition of core functions of QC class

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/data_out_atom_data.h>
#include <deal.II-qc/base/quadrature_lib.h>
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;


template <int dim, typename PotentialType, int atomicity>
QC<dim, PotentialType, atomicity>::~QC ()
{
  dof_handler.clear();
}



template <int dim, typename PotentialType, int atomicity>
QC<dim, PotentialType, atomicity>::QC (const ConfigureQC &config)
  :
#ifdef DEAL_II_TRILINOS_WITH_ROL
  qc_objective(*this),
#endif
  mpi_communicator(MPI_COMM_WORLD),
  this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator)),
  n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
  pcout (std::cout, this_mpi_process==0),
  configure_qc (config),
  triangulation (mpi_communicator,
                 // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                 Triangulation<dim, spacedim>::limit_level_difference_at_vertices,
                 configure_qc.get_ghost_cell_layer_thickness()),
  fe (FE_Q<dim, spacedim>(1), dim *atomicity),
  u_fe (),
  dof_handler (triangulation),
  molecule_handler (configure_qc),
  computing_timer (mpi_communicator,
                   pcout,
                   TimerOutput::never,
                   TimerOutput::wall_times)
{
  Assert (dim==configure_qc.get_dimension(), ExcInternalError());

  for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
    u_fe[atom_stamp] = atom_stamp*dim;

  // Load the mesh by reading from mesh file
  setup_triangulation();

  // Read atom data file and initialize atoms
  setup_cell_molecules();

  // Initialize boundary functions.
  initialize_boundary_functions();
}



template <int dim, typename PotentialType, int atomicity>
void
QC<dim, PotentialType, atomicity>::run (const bool relaxed_configuration_as_reference)
{
  setup_cell_energy_molecules();
  setup_system();
  setup_fe_values_objects();
  update_neighbor_lists();
  update_positions();

  // Molecular statics relaxation.
  minimize_energy (-1.);

  if (relaxed_configuration_as_reference)
    // Measure displacement taking the relaxed configuration as reference.
    for (auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
      for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
        cell_molecule.second.atoms[atom_stamp].initial_position =
          cell_molecule.second.atoms[atom_stamp].position;

  // Initialize external potential fields.
  initialize_external_potential_fields();

  const unsigned int n_time_steps = configure_qc.get_n_time_steps();
  const double time_step   = configure_qc.get_time_step();

  double time = 0.;
  for (unsigned int step = 0; step <= n_time_steps; ++step)
    {
      minimize_energy(time);
      time += time_step;
      output_results (time, step);
    }
}



template <int dim, typename PotentialType, int atomicity>
void
QC<dim, PotentialType, atomicity>::
output_results (const double time,
                const unsigned int timestep_no) const
{
  const std::string &atom_data_name = "atom_data_";

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (atomicity*dim,
                  DataComponentInterpretation::component_is_part_of_vector);

  dealiiqc::Utilities::write_vector_out (locally_relevant_displacement,
                                         dof_handler,
                                         "solution_",
                                         time,
                                         timestep_no,
                                         interpretation);

  AssertThrow (n_mpi_processes < 1000,
               ExcNotImplemented());

  const std::string atom_data_filename =
    dealiiqc::Utilities::data_out_filename (atom_data_name,
                                            timestep_no,
                                            this_mpi_process,
                                            ".vtp");
  std::ofstream atom_data_output(atom_data_filename.c_str());

  dealii::DataOutBase::VtkFlags flags (std::numeric_limits<double>::min(),
                                       std::numeric_limits<unsigned int>::min(),
                                       false);

  DataOutAtomData atom_data_out;
  atom_data_out.write_vtp<dim> (cell_molecule_data.cell_energy_molecules,
                                flags,
                                atom_data_output);

  if (this_mpi_process==0)
    {
      std::vector<std::string> atom_data_filenames;
      for (unsigned int i=0; i<n_mpi_processes; ++i)
        atom_data_filenames.push_back
        (dealiiqc::Utilities::data_out_filename (atom_data_name,
                                                 timestep_no,
                                                 i,
                                                 ".vtp"));

      const std::string
      pvtp_atom_data_master_filename = (atom_data_name +
                                        dealii::Utilities::int_to_string(timestep_no,4) +
                                        ".pvtp");

      std::ofstream pvtp_atom_data_master (pvtp_atom_data_master_filename.c_str());

      atom_data_out.write_pvtp_record (atom_data_filenames,
                                       flags,
                                       pvtp_atom_data_master);

      static std::vector<std::pair<double, std::string> >
      times_and_atom_data_names;

      times_and_atom_data_names.push_back(std::make_pair(time,
                                                         pvtp_atom_data_master_filename.c_str()));
      std::ofstream pvd_atom_data_output (atom_data_name + ".pvd");
      DataOutBase::write_pvd_record (pvd_atom_data_output,
                                     times_and_atom_data_names);
    }
}



template <int dim, typename PotentialType, int atomicity>
void
QC<dim, PotentialType, atomicity>::reconfigure_qc(const ConfigureQC &configure)
{
  configure_qc = configure;
  setup_cell_energy_molecules();
  setup_system();
  setup_fe_values_objects();
  update_neighbor_lists();
  update_positions();
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_triangulation()
{
  configure_qc.get_geometry<dim>()->create_mesh(triangulation);

  // --- Perform a-priori refinement.

  ConfigureQC::InitialRefinementParameters initial_refinement_params =
    configure_qc.get_initial_refinement_parameters();

  FunctionParser<spacedim> refinement_function (1,0.);

  refinement_function.initialize(FunctionParser<spacedim>::default_variable_names(),
                                 initial_refinement_params.indicator_function,
                                 typename FunctionParser<spacedim>::ConstMap());

  const double refinement_parameter = initial_refinement_params.refinement_parameter;

  const unsigned int n_refinement_cycles =
    initial_refinement_params.n_refinement_cycles;

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; cycle++)
    {
      Vector<float> blind_error_estimate_per_cell (triangulation.n_active_cells());

      unsigned int active_cell_i = 0;
      for (types::CellIteratorType<dim, spacedim>
           cell  = triangulation.begin_active();
           cell != triangulation.end();
           cell++, active_cell_i++)
        blind_error_estimate_per_cell[active_cell_i] =
          refinement_function.value(cell->center());

      // FIXME: Support other ways to mark cells.
      GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                       blind_error_estimate_per_cell,
                                                       refinement_parameter, 0);
      triangulation.execute_coarsening_and_refinement();
    }
  triangulation.setup_ghost_cells();
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_cell_molecules()
{
  TimerOutput::Scope t (computing_timer, "Parse and assign all atoms to cells");

  const std::string atom_data_file = configure_qc.get_atom_data_file();

  if (!atom_data_file.empty())
    {
      std::fstream fin(atom_data_file, std::fstream::in);
      cell_molecule_data =
        CellMoleculeTools::build_cell_molecule_data<dim, atomicity, spacedim>
        (fin, triangulation);
    }
  else if (!configure_qc.get_stream()->eof())
    cell_molecule_data =
      CellMoleculeTools::build_cell_molecule_data<dim, atomicity, spacedim>
      (*configure_qc.get_stream(), triangulation);
  else
    AssertThrow(false,
                ExcMessage("Atom data was not provided neither as an auxiliary "
                           "data file nor at the end of the parameter file!"));

  // It is ConfigureQC that actually creates a PotentialType object according
  // to the parsed input and can return a shared pointer to the PotentialType
  // object. However, charges in PotentialType object aren't set yet.
  // Finish setting up PotentialType object here.
  configure_qc.get_potential()->set_charges(cell_molecule_data.charges);
}

template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_cell_energy_molecules()
{
  TimerOutput::Scope t (computing_timer,
                        "Setup energy molecules with cluster weights");

  // It is ConfigureQC that actually creates a shared pointer to the derived
  // class object of the Cluster::WeightsByBase according to the parsed input.

  cluster_weights_method = configure_qc.get_cluster_weights<dim, atomicity, spacedim>();

  // Get Quadrature from ConfigureQC.
  if (configure_qc.get_quadrature_rule() == "QTrapezWithMidpoint")
    cluster_weights_method->initialize (triangulation,
                                        QTrapezWithMidpoint<dim>());
  else if (configure_qc.get_quadrature_rule() == "QTrapez")
    cluster_weights_method->initialize (triangulation,
                                        QTrapez<dim>());
  else
    AssertThrow (false, ExcNotImplemented());

  cell_molecule_data.cell_energy_molecules =
    cluster_weights_method->
    update_cluster_weights (triangulation,
                            cell_molecule_data.cell_molecules);
}



template <int dim, typename PotentialType, int atomicity>
template<typename T>
void QC<dim, PotentialType, atomicity>::write_mesh (T &os, const std::string &type )
{
  GridOut grid_out;
  if ( !type.compare("eps")  )
    grid_out.write_eps (triangulation, os);
  else if ( !type.compare("msh") )
    grid_out.write_msh (triangulation, os);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::initialize_boundary_functions()
{
  TimerOutput::Scope t (computing_timer, "Initialize boundary functions");

  std::map<unsigned int, std::vector<std::string> >
  boundary_ids_to_function_expressions = configure_qc.get_boundary_functions();

  for (auto &single_bc : boundary_ids_to_function_expressions)
    {
      const unsigned int n_components = single_bc.second.size();

      Assert (n_components == dim * atomicity,
              ExcMessage("Invalid number of components."));

      std::vector<bool> component_mask (n_components, true);

      for (unsigned int i = 0; i < n_components; ++i)
        if (single_bc.second[i].empty())
          {
            component_mask[i] = false;
            // Set empty strings to parsable strings.
            // Entry in component mask will ensure that this is not used at all.
            single_bc.second[i] = "0.";
          }

      dirichlet_boundary_functions.insert
      (std::make_pair (single_bc.first,
                       std::make_pair (component_mask,
                                       std::make_shared<FunctionParser<spacedim> >(dim*atomicity, 0.)))
      );

      dirichlet_boundary_functions[single_bc.first].second->
      initialize ((spacedim==3) ? "x,y,z,t" :
                  (spacedim==2  ? "x,y,t"   : "x,t"),
                  single_bc.second,
                  typename FunctionParser<spacedim>::ConstMap(),
                  true);
    }
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_boundary_conditions (const double time)
{
  TimerOutput::Scope t (computing_timer, "Setup boundary conditions");

  constraints.clear();
  constraints.reinit (locally_relevant_set);
  constraints.merge(hanging_node_constraints);

  for (auto &single_bc : dirichlet_boundary_functions)
    single_bc.second.second->set_time(time);

  for (const auto &single_bc : dirichlet_boundary_functions)
    VectorTools::interpolate_boundary_values (dof_handler,
                                              single_bc.first,
                                              *(single_bc.second.second),
                                              constraints,
                                              single_bc.second.first);

  constraints.close ();

}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::initialize_external_potential_fields (const double initial_time)
{
  for (const auto &entry : configure_qc.get_external_potential_fields())
    {
      auto external_potential_field_iterator =
        external_potential_fields.insert
        (
          std::make_pair(entry.first.first,
                         std::make_shared<PotentialFieldFunctionParser<spacedim> >
                         (entry.first.second,
                          initial_time))
        );

      // Initialize FunctionParser object of PotentialFieldParser.
      static_cast<PotentialFieldFunctionParser<spacedim> *>
      (external_potential_field_iterator->second.get())->
      initialize ((spacedim==3) ? "x,y,z,t" :
                  (spacedim==2  ? "x,y,t"   : "x,t"),
                  entry.second,
                  typename FunctionParser<spacedim>::ConstMap(),
                  true);
    }
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_system ()
{
  TimerOutput::Scope t (computing_timer, "Setup system");

  dof_handler.distribute_dofs (fe);

  // Define blocks indices.
  // Here block index is same as atom_stamp
  std::vector<unsigned int> component_to_block_indices (dim*atomicity, 0);

  for (int i = 0; i < dim*atomicity; ++i)
    component_to_block_indices[i] = std::div(i, dim).quot;

  // Renumber DoFs block-wise.
  DoFRenumbering::component_wise (dof_handler, component_to_block_indices);

  std::vector<dealii::types::global_dof_index> n_dofs_per_block (atomicity);

  // Prepare number of dofs per block
  DoFTools::count_dofs_per_block (dof_handler,
                                  n_dofs_per_block,
                                  component_to_block_indices);

  // All blocks should have equal number of DoFs.
  Assert (atomicity > 1 ? std::equal(n_dofs_per_block.begin()+1, // Check from the second element.
                                     n_dofs_per_block.end(),
                                     n_dofs_per_block.begin())
          : true,
          ExcInternalError());


  std::vector<IndexSet> locally_owned_partitioning   (atomicity);
  std::vector<IndexSet> locally_relevant_partitioning(atomicity);

  {
    const IndexSet locally_owned_set = dof_handler.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_set);

    for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
      {
        locally_owned_partitioning[atom_stamp]    = locally_owned_set.get_view
                                                    (n_dofs_per_block[0]*(atom_stamp  ),
                                                     n_dofs_per_block[0]*(atom_stamp+1));
        locally_relevant_partitioning[atom_stamp] = locally_relevant_set.get_view
                                                    (n_dofs_per_block[0]*(atom_stamp  ),
                                                     n_dofs_per_block[0]*(atom_stamp+1));
      }
  }

  // set-up constraints objects
  hanging_node_constraints.reinit (locally_relevant_set);
  DoFTools::make_hanging_node_constraints (dof_handler, hanging_node_constraints);
  hanging_node_constraints.close ();

  // Merging with `constraints` is faster if hanging_node_constraints is closed.
  setup_boundary_conditions (0./*initial time*/);

  distributed_displacement.reinit (locally_owned_partitioning,
                                   mpi_communicator);

  locally_relevant_displacement.reinit (locally_owned_partitioning,
                                        locally_relevant_partitioning,
                                        mpi_communicator,
                                        false);

  locally_relevant_gradient.reinit (locally_owned_partitioning,
                                    locally_relevant_partitioning,
                                    mpi_communicator,
                                    true);

  distributed_displacement      = 0.;
  locally_relevant_displacement = 0.;
  locally_relevant_gradient     = 0.;

  // Create a temporary vector to initialize inverse_mass_matrix;
  vector_t inverse_masses (locally_owned_partitioning,
                           locally_relevant_partitioning,
                           mpi_communicator,
                           true);

  // Compute inverse masses based on cluster weights.
  // FIXME: Skip preparation if OptimalSummationRules is used.
  if (std::dynamic_pointer_cast<Cluster::WeightsByOptimalSummationRules<dim> >
      (cluster_weights_method) == NULL)
    cluster_weights_method->compute_dof_inverse_masses (inverse_masses,
                                                        cell_molecule_data,
                                                        dof_handler,
                                                        hanging_node_constraints);
  inverse_mass_matrix.reinit (inverse_masses);

  cells_to_data.clear();

  // TODO: use TriaAccessor<>::set_user_pointer() to associate AssemblyData with a cell
  for (types::DoFCellIteratorType<dim, spacedim>
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    if (!cell->is_artificial())
      cells_to_data.insert (std::make_pair (cell,
                                            AssemblyData()));
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::setup_fe_values_objects ()
{
  TimerOutput::Scope t (computing_timer, "Setup FEValues objects");

  // Container to store quadrature points and weights.
  std::vector<Point<dim>> points;
  std::vector<double> weights_per_atom;

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Get a non-constant reference to energy molecules to update
  // local index (within a cell) of energy molecules.
  auto &cell_energy_molecules = cell_molecule_data.cell_energy_molecules;

  for (types::DoFCellIteratorType<dim, spacedim>
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    {
      // Loop through locally relevant cells.
      if (cell->is_artificial())
        continue;

      // Include all the energy molecules associated to this active cell
      // as quadrature points. The quadrature points will be then used to
      // initialize fe_values object so as to evaluate the displacement
      // at the initial location of energy molecules in the active cells.

      const auto energy_molecules_range =
        CellMoleculeTools::molecules_range_in_cell<dim, atomicity, spacedim>
        (cell, cell_energy_molecules);

      const unsigned int n_energy_molecules_in_cell =
        energy_molecules_range.second;

      // If this cell is not within the locally relevant active cells of the
      // current MPI process continue active cell loop
      if (n_energy_molecules_in_cell == 0)
        continue;

      // Resize containers to known number of energy molecules in cell.
      points.resize(n_energy_molecules_in_cell);
      weights_per_atom.resize(n_energy_molecules_in_cell);

      AssemblyData &data = cells_to_data[cell];

      // We need non-const iterator to update local index of energy molecule.
      // TODO: Move the task of updating local index of energy atoms to
      //       somewhere else? For now keeping it here.
      // To get a non-const iterator to the beginning of energy atom range
      // call erase with the same argument. Strictly no erase is performed
      // but erase would return a non-const iterator.
      // This is not exactly a hack as we are using how erase on containers
      // must behave. Also, this is a constant time operation.
      types::CellMoleculeIteratorType<dim, atomicity, spacedim>
      cell_energy_molecule_iterator =
        cell_energy_molecules.erase(energy_molecules_range.first.first,
                                    energy_molecules_range.first.first);

      for (unsigned int q = 0; q < n_energy_molecules_in_cell; ++q, ++cell_energy_molecule_iterator)
        {
          // const_iter->second yields the actual atom
          for (int d=0; d<dim; ++d)
            // Copying only dim-dimensions.
            points[q][d]      = cell_energy_molecule_iterator->second.position_inside_reference_cell[d];

          weights_per_atom[q] = cell_energy_molecule_iterator->second.cluster_weight;
          cell_energy_molecule_iterator->second.local_index = q;
        }

      Assert (cell_energy_molecule_iterator == energy_molecules_range.first.second,
              ExcMessage("The number of energy molecule in the cell counted "
                         "using the distance between the iterator ranges "
                         "yields a different result than "
                         "incrementing the iterator to cell_energy_molecules."
                         "Why wasn't this error thrown earlier?"));

      Assert (data.fe_values.use_count() ==0,
              ExcInternalError());

      // Now we are ready to initialize FEValues object.
      data.fe_values =
        std::make_shared<FEValues<dim, spacedim>> (mapping, fe,
                                                   Quadrature<dim> (points,
                                                       weights_per_atom),
                                                   update_values);

      // finally reinit FEValues so that it's ready to provide all required
      // information:
      data.fe_values->reinit(cell);

      for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
        data.displacements[atom_stamp].resize(points.size());
    }
}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::update_positions()
{
  TimerOutput::Scope t (computing_timer, "Update energy molecules' positions");

  // Loop over all the locally relevant cells and evaluate displacement field at
  // quadrature points.
  // This is needed irrespectively of energy or gradient calculations.
  for (auto
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    {
      if (cell->is_artificial())
        continue;

      const auto it = cells_to_data.find(cell);
      Assert (it != cells_to_data.end(),
              ExcInternalError());

      auto &data = it->second;

      std::pair<types::CellMoleculeIteratorType<dim, atomicity, spacedim>, types::CellMoleculeIteratorType<dim, atomicity, spacedim> >
      cell_energy_molecules_range = cell_molecule_data.cell_energy_molecules.equal_range(cell);

#ifdef DEBUG
      const unsigned int
      n_energy_molecules_in_cell = cell_molecule_data.cell_energy_molecules.count(cell);
      for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
        // Size of the displacements should be same as the total number of the
        // energy molecules on a per cell basis.
        Assert (data.displacements[atom_stamp].size()==n_energy_molecules_in_cell,
                ExcInternalError());
#endif

      // If this locally relevant active cell doesn't contain any atoms
      // continue active cell loop
      if (cell_energy_molecules_range.first==cell_energy_molecules_range.second)
        continue;

      for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
        // Get displacement field for each atom_stamp at all quadrature points
        // of this cell.
        (*data.fe_values)[u_fe[atom_stamp]].get_function_values(locally_relevant_displacement,
                                                                data.displacements[atom_stamp]);
      const auto &displacements = data.displacements;

      // Update energy molecules positions.
      for (unsigned int i = 0;
           cell_energy_molecules_range.first != cell_energy_molecules_range.second;
           cell_energy_molecules_range.first++, i++)
        for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
          for (int d=0; d<dim; ++d)
            // Copy only dim-dimensions.
            cell_energy_molecules_range.first->second.atoms[atom_stamp].position[d] =
              cell_energy_molecules_range.first->second.atoms[atom_stamp].initial_position[d] +
              displacements[atom_stamp][i][d];

      // The loop over displacements must have exhausted all the energy
      // molecules on a per cell basis (and the converse also should be true).
      Assert (cell_energy_molecules_range.first == cell_energy_molecules_range.second,
              ExcInternalError());
    }

}



template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::update_neighbor_lists()
{
  TimerOutput::Scope t (computing_timer, "Update neighbor lists");

  // TODO: Update neighbor lists
  // if( (iter_count % neigh_modify_delay)==0 || (max_abs_displacement > neigh_skin)   )
  //   update_neighbour_lists();
  neighbor_lists = molecule_handler.get_neighbor_lists(cell_molecule_data.cell_energy_molecules);
}



template <int dim, typename PotentialType, int atomicity>
template <bool ComputeGradient>
double QC<dim, PotentialType, atomicity>::compute (vector_t &gradient) const
{
  TimerOutput::Scope t (computing_timer, "Compute energy and gradient");

  if (ComputeGradient)
    gradient = 0.;

  const double energy_per_process =
    neighbor_lists.empty() ?
    0.                     :
    compute_local<ComputeGradient>(gradient);

  gradient.compress(VectorOperation::add);

  // sum contributions from all MPI cores and return the result:
  return dealii::Utilities::MPI::sum(energy_per_process, mpi_communicator);
}



template <int dim, typename PotentialType, int atomicity>
template <bool ComputeGradient>
double QC<dim, PotentialType, atomicity>::compute_local (vector_t &gradient) const
{
  double energy_per_process = 0.;

  // Get the const PotentialType object from configure_qc.
  const std::shared_ptr<const PotentialType> potential_ptr =
    std::const_pointer_cast<const PotentialType>(
      std::static_pointer_cast<PotentialType>(configure_qc.get_potential()));

  // Assert that the casted pointer is not NULL.
  Assert (potential_ptr, ExcInternalError());

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<dealii::types::global_dof_index>
  local_dof_indices_I(dofs_per_cell), local_dof_indices_J(dofs_per_cell);
  Vector<double> local_gradient_I(dofs_per_cell), local_gradient_J(dofs_per_cell);

  // start from a first pair of cells I-J in the neighbour list.
  const types::CellIteratorType<dim, spacedim>
  cell_I_first = neighbor_lists.begin()->first.first,
  cell_J_first = neighbor_lists.begin()->first.second;

  // Convert tria's cells into dof cells.
  const types::DoFCellIteratorType<dim, spacedim>
  dof_cell_I_first (&triangulation,
                    cell_I_first->level(),
                    cell_I_first->index(),
                    &dof_handler);

  const types::DoFCellIteratorType<dim, spacedim>
  dof_cell_J_first (&triangulation,
                    cell_J_first->level(),
                    cell_J_first->index(),
                    &dof_handler);

  // Locate the first pair of cells I-J.
  typename
  std::map<types::DoFCellIteratorType<dim, spacedim>, AssemblyData>::const_iterator
  cell_data_I = cells_to_data.find(dof_cell_I_first),
  cell_data_J = cells_to_data.find(dof_cell_J_first);

  AssertThrow (cell_data_I != cells_to_data.end() &&
               cell_data_J != cells_to_data.end(),
               ExcInternalError());

  local_gradient_I = 0.;
  local_gradient_J = 0.;

  cell_data_I->first->get_dof_indices (local_dof_indices_I);
  cell_data_J->first->get_dof_indices (local_dof_indices_J);

  // TODO: parallelize using using TBB by looping over pairs of cells.
  // Rework neighbor list as std::map<std::pair<Cell,Cell>,....>
  for (const auto &cell_pair_cell_molecule_pair : neighbor_lists)
    {
      // get reference to current cell pair and molecule pair
      const types::CellIteratorType<dim, spacedim>
      &cell_I  = cell_pair_cell_molecule_pair.first.first,
       &cell_J = cell_pair_cell_molecule_pair.first.second;

      const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      &cell_molecule_I  = cell_pair_cell_molecule_pair.second.first,
       &cell_molecule_J = cell_pair_cell_molecule_pair.second.second;

      Assert ((cell_I == cell_molecule_I->first) &&
              (cell_J == cell_molecule_J->first),
              ExcMessage("Incorrect neighbor lists."
                         "Either cell_I or cell_J doesn't contain "
                         "cell_molecule_I or cell_molecule_J, respectively."));

      const Molecule<spacedim, atomicity> &molecule_I = cell_molecule_I->second;
      const Molecule<spacedim, atomicity> &molecule_J = cell_molecule_J->second;

      // Molecules I and J interacting with each other could be belong to
      // different clusters, and therefore have different cluster weights.
      // In such a case, we need to account for different weights associated
      // with the molecules by scaling E_{IJ} with (n_I + n_J)/2,
      // which is exactly how this contribution would be added had we followed
      // assembly from clusters perspective.
      // Since molecules not attributed to clusters have zero weights,
      // we can directly use the scaling above without the need to find out
      // whether or not one of the two molecules do not belong to any cluster.

      // For OptimalSummationRules, if molecule_J is not a sampling molecule
      // then E_{IJ} is scaled by just n_I (the sampling weight of molecule_I).

      const bool optimal_summation_rule_used =
        std::dynamic_pointer_cast
        <Cluster::WeightsByOptimalSummationRules<dim, atomicity, spacedim> >
        (cluster_weights_method) != NULL;

      // Compute scaling of energy due to cluster weights.
      const double scale_energy = (optimal_summation_rule_used &&
                                   molecule_J.cluster_weight==0)
                                  ?
                                  molecule_I.cluster_weight
                                  :
                                  0.5 * (molecule_I.cluster_weight +
                                         molecule_J.cluster_weight );

      // Assert that the scaling factor of energy, due to clustering, is
      // non-zero. One of the molecule must be a cluster molecule
      // (more specifically molecule I must be cluster molecule) and
      // therefore have non-zero (more specifically positive) cluster_weight.
      Assert (scale_energy > 0, ExcInternalError());

      // TODO: Tensors should be dim-dimensional. Following code needs to be
      // adjusted and ComputeTools:: functions as well.
      Assert (spacedim==dim, ExcNotImplemented());

      // 1 --- Inter-molecular contribution from molecule I and J interactions.
      const std::pair<double, Table<2, Tensor<1, dim> > >
      molecular_IJ =
        ComputeTools::energy_and_gradient
        <PotentialType, dim, atomicity, ComputeGradient> (*potential_ptr,
                                                          molecule_I,
                                                          molecule_J);

      // 2 --- Intra-molecular contribution from molecule I.
      const std::pair<double, std::array<Tensor<1, dim>, atomicity > >
      intra_I =
        ComputeTools::energy_and_gradient
        <PotentialType, dim, atomicity, ComputeGradient> (*potential_ptr,
                                                          molecule_I);

      // 3 --- Intra-molecular contribution from molecule J.
      const std::pair<double, std::array<Tensor<1, dim>, atomicity > >
      intra_J =
        molecule_J.cluster_weight > 0
        ?
        ComputeTools::energy_and_gradient
        <PotentialType, dim, atomicity, ComputeGradient> (*potential_ptr,
                                                          molecule_J)
        :
        std::make_pair (0., std::array<Tensor<1, dim>, atomicity>());;

      // 4 --- Contribution due to external potential field on molecule I.
      std::pair<double, std::array<Tensor<1, dim>, atomicity> >
      external_I =
        ComputeTools::energy_and_gradient
        <dim, atomicity, ComputeGradient> (external_potential_fields,
                                           cell_I->material_id(),
                                           molecule_I,
                                           *cell_molecule_data.charges);

      // 5 --- Contribution due to external potential field on molecule J.
      std::pair<double, std::array<Tensor<1, dim>, atomicity> >
      external_J =
        molecule_J.cluster_weight > 0
        ?
        ComputeTools::energy_and_gradient
        <dim, atomicity, ComputeGradient> (external_potential_fields,
                                           cell_J->material_id(),
                                           molecule_J,
                                           *cell_molecule_data.charges)
        :
        std::make_pair (0., std::array<Tensor<1, dim>, atomicity>());

      // Now we scale the energy according to cluster weights of the molecules.
      energy_per_process +=
        molecular_IJ.first                           * scale_energy    +
        (intra_I.first + external_I.first) * molecule_I.cluster_weight +
        (intra_J.first + external_J.first) * molecule_J.cluster_weight;

      if (ComputeGradient)
        {
          const types::DoFCellIteratorType<dim, spacedim>
          dof_cell_I (&triangulation,
                      cell_I->level(),
                      cell_I->index(),
                      &dof_handler);

          const types::DoFCellIteratorType<dim, spacedim>
          dof_cell_J (&triangulation,
                      cell_J->level(),
                      cell_J->index(),
                      &dof_handler);

          // Check if I'th or Jth cell changed, if so, update the pointer to
          // the cell data and get dof indices.
          if (cell_data_I->first != dof_cell_I)
            {
              // FIXME: this is quite awkward, as we need to flash-out
              // local contributions here
              constraints.distribute_local_to_global(local_gradient_I,
                                                     local_dof_indices_I,
                                                     gradient);

              cell_data_I = cells_to_data.find(dof_cell_I);
              dof_cell_I->get_dof_indices (local_dof_indices_I);
              local_gradient_I = 0.;
            }
          if (cell_data_J->first != dof_cell_J)
            {
              constraints.distribute_local_to_global(local_gradient_J,
                                                     local_dof_indices_J,
                                                     gradient);

              cell_data_J = cells_to_data.find(dof_cell_J);
              dof_cell_J->get_dof_indices (local_dof_indices_J);
              local_gradient_J = 0.;
            }

          // Get quadrature point index from the local_index of molecule pairs
          const unsigned int
          qI = molecule_I.local_index,
          qJ = molecule_J.local_index;

          std::array<Tensor<1, dim>, atomicity>  gradient_I, gradient_J;

          Tensor<1, dim> molecular_IJ_i; // Temporary variable.
          Tensor<1, dim> molecular_IJ_j; // Temporary variable.
          for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
            {
              molecular_IJ_i = 0.;
              molecular_IJ_j = 0.;
              for (int i = 0; i < atomicity; ++i)
                {
                  // Yields summation over the rows of the Table.
                  molecular_IJ_i  +=      molecular_IJ.second(atom_stamp, i);

                  // Yields summation over the columns of the Table.
                  molecular_IJ_j  +=      molecular_IJ.second(i, atom_stamp);
                }

              gradient_I[atom_stamp] =
                molecular_IJ_i                    * scale_energy +
                (intra_I.   second[atom_stamp] +
                 external_I.second[atom_stamp]  ) * molecule_I.cluster_weight;

              gradient_J[atom_stamp] =
                -molecular_IJ_j                   *  scale_energy +
                (intra_J.   second[atom_stamp] +
                 external_J.second[atom_stamp]  ) * molecule_J.cluster_weight;
            }

          // Finally, we evaluated local contribution to the gradient of
          // energy. The main ingredient in forces is
          // r^{ab}_{,k} = n^{ab} * [N_k(X^a) - N_k(X^b)]
          // where k is global dof. So for a given pair of atoms a and b,
          // we can add force contributions from
          // F(local_dofs(i)) +=  n^{ab}*N_{local_dof{i}}(X^a)
          // F(local_dofs(j)) -=  n^{ab}*N_{local_dof{j}}(X^b)
          // Below we utilize the fact that we work with primitive vector-valued
          // shape functions which are non-zero at a single component only.
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const unsigned int component_index = fe.system_to_component_index(k).first;

              const unsigned int atom_stamp   = std::div(component_index, dim).quot;
              const unsigned int nonzero_comp = component_index % dim;

              local_gradient_I[k] += gradient_I[atom_stamp][nonzero_comp]
                                     *
                                     cell_data_I->second.fe_values->shape_value(k, qI);

              local_gradient_J[k] += gradient_J[atom_stamp][nonzero_comp]
                                     *
                                     cell_data_J->second.fe_values->shape_value(k, qJ);
            }
        }

    } // loop over neighbour lists

  // Flush out contributions from the last pair of cells
  constraints.distribute_local_to_global(local_gradient_I,
                                         local_dof_indices_I,
                                         gradient);
  constraints.distribute_local_to_global(local_gradient_J,
                                         local_dof_indices_J,
                                         gradient);
  return energy_per_process;
}

template <int dim, typename PotentialType, int atomicity>
void QC<dim, PotentialType, atomicity>::minimize_energy (const double time)
{
  if (time >= 0.)
    {
      for (auto &potential_field : external_potential_fields)
        potential_field.second->set_time (time);

      setup_boundary_conditions(time);
    }

  //---------------------------------------------------------------------------

  std::function<double(vector_t &,  const vector_t &)> compute_function =
    [&]               (vector_t &G, const vector_t &) -> double
  {
    constraints.distribute(distributed_displacement);
    locally_relevant_displacement = distributed_displacement;
    constraints.set_zero(distributed_displacement);
    update_positions();
    update_neighbor_lists();
    const double energy = compute<true>(locally_relevant_gradient);
    G = locally_relevant_gradient;
    return energy;
  };

  ConfigureQC::SolverControlParameters solver_control_parameters =
    configure_qc.get_solver_control_parameters();

  SolverControl solver_control (solver_control_parameters.max_steps,
                                solver_control_parameters.tolerance,
                                solver_control_parameters.log_history,
                                solver_control_parameters.log_result);
  solver_control.log_frequency(solver_control_parameters.log_frequency);

  try
    {
      if (configure_qc.get_minimizer_name()=="FIRE")
        {
          ConfigureQC::FireParameters fire_parameters =
            configure_qc.get_fire_parameters();

          typename SolverFIRE<vector_t>::AdditionalData
          additional_data_fire (fire_parameters.initial_time_step,
                                fire_parameters.maximum_time_step,
                                fire_parameters.maximum_linfty_norm);

          SolverFIRE<vector_t> solver(solver_control, additional_data_fire);
          solver.solve (compute_function,
                        distributed_displacement,
                        inverse_mass_matrix);
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
  catch (...)
    {
      pcout << "Solver terminated without achieving desired tolerance, "
            << "while minimizing the energy to "
            << compute<false>(locally_relevant_gradient)
            << " before terminating."
            << std::endl;
    }
}


// Instantiations
#define SINGLE_QC_INSTANTIATION(_DIM, _POTENTIAL, _ATOMICITY) \
  template class QC<_DIM, _POTENTIAL, _ATOMICITY>; \
  template double QC<_DIM, _POTENTIAL, _ATOMICITY>::compute<true>(dealii::TrilinosWrappers::MPI::BlockVector&) const; \
  template double QC<_DIM, _POTENTIAL, _ATOMICITY>::compute<false>(dealii::TrilinosWrappers::MPI::BlockVector&) const; \
  template double QC<_DIM, _POTENTIAL, _ATOMICITY>::compute_local<true>(dealii::TrilinosWrappers::MPI::BlockVector&) const; \
  template double QC<_DIM, _POTENTIAL, _ATOMICITY>::compute_local<false>(dealii::TrilinosWrappers::MPI::BlockVector&) const;

#define QC_INSTANTIATIONS(R, X)  \
  SINGLE_QC_INSTANTIATION(BOOST_PP_TUPLE_ELEM(2, 0, X), Potential::PairLJCutManager,    BOOST_PP_TUPLE_ELEM(2, 1, X)) \
  SINGLE_QC_INSTANTIATION(BOOST_PP_TUPLE_ELEM(2, 0, X), Potential::PairCoulWolfManager, BOOST_PP_TUPLE_ELEM(2, 1, X)) \
  SINGLE_QC_INSTANTIATION(BOOST_PP_TUPLE_ELEM(2, 0, X), Potential::PairLJCutCoulWolfManager, BOOST_PP_TUPLE_ELEM(2, 1, X))

BOOST_PP_LIST_FOR_EACH_PRODUCT(QC_INSTANTIATIONS, 2, (DIM, ATOMICITY))


DEAL_II_QC_NAMESPACE_CLOSE
