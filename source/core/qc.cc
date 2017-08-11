
// a source file which contains definition of core functions of QC class

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/core/qc.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;


template <int dim, typename PotentialType>
QC<dim, PotentialType>::~QC ()
{
  dof_handler.clear();
}



template <int dim, typename PotentialType>
QC<dim, PotentialType>::QC (const ConfigureQC &config)
  :
  mpi_communicator(MPI_COMM_WORLD),
  pcout (std::cout,
         (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
          == 0)),
  configure_qc (config),
  triangulation (mpi_communicator,
                 // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                 Triangulation<dim>::limit_level_difference_at_vertices),
  fe (FE_Q<dim>(1),dim),
  u_fe (0),
  dof_handler (triangulation),
  molecule_handler (configure_qc),
  computing_timer (mpi_communicator,
                   pcout,
                   TimerOutput::never,
                   TimerOutput::wall_times)
{
  Assert (dim==configure_qc.get_dimension(), ExcInternalError());

  // Load the mesh by reading from mesh file
  setup_triangulation();

  // Read atom data file and initialize atoms
  setup_cell_molecules();

  // Initialize boundary functions.
  initialize_boundary_functions();
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::run ()
{
  setup_cell_energy_molecules();
  setup_system();
  setup_fe_values_objects();
  update_neighbor_lists();
  update_positions();

  minimize_energy (-1.);

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



namespace
{
  inline
  std::string data_out_solution_filename (const unsigned int timestep_no,
                                          const dealii::types::subdomain_id id)
  {
    return "solution-" +
           dealii::Utilities::int_to_string(timestep_no,4) + "." +
           dealii::Utilities::int_to_string(id,3) + ".vtu";
  }
}

template <int dim, typename PotentialType>
void QC<dim, PotentialType>::output_results (const double time,
                                             const unsigned int timestep_no) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  std::vector<std::string> solution_names;
  switch (dim)
    {
    case 1:
      solution_names.push_back ("u_x");
      break;
    case 2:
      solution_names.push_back ("u_x");
      solution_names.push_back ("u_y");
      break;
    case 3:
      solution_names.push_back ("u_x");
      solution_names.push_back ("u_y");
      solution_names.push_back ("u_z");
      break;
    default:
      Assert (false, ExcNotImplemented());
    }
  data_out.add_data_vector (locally_relevant_displacement,
                            solution_names);

  std::vector<dealii::types::subdomain_id> partition_int (triangulation.n_active_cells());
  GridTools::get_subdomain_association (triangulation, partition_int);
  const Vector<float> partitioning (partition_int.begin(),
                                    partition_int.end());
  data_out.add_data_vector (partitioning, "partitioning");
  data_out.build_patches ();

  const unsigned int
  this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator),
  n_mpi_processes  = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

  AssertThrow (n_mpi_processes < 1000,
               ExcNotImplemented());

  const std::string filename = data_out_solution_filename (timestep_no,
                                                           this_mpi_process);

  std::ofstream output (filename.c_str());
  data_out.write_vtu (output);
  if (this_mpi_process==0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0; i<n_mpi_processes; ++i)
        filenames.push_back (data_out_solution_filename (timestep_no, i));

      const std::string
      visit_master_filename = ("solution-" +
                               dealii::Utilities::int_to_string(timestep_no,4) +
                               ".visit");
      std::ofstream visit_master (visit_master_filename.c_str());
      DataOutBase::write_visit_record (visit_master, filenames);
      const std::string
      pvtu_master_filename = ("solution-" +
                              dealii::Utilities::int_to_string(timestep_no,4) +
                              ".pvtu");
      std::ofstream pvtu_master (pvtu_master_filename.c_str());
      data_out.write_pvtu_record (pvtu_master, filenames);
      static std::vector<std::pair<double,std::string> > times_and_names;
      times_and_names.push_back (std::pair<double,std::string> (time,
                                                                pvtu_master_filename));
      std::ofstream pvd_output ("solution.pvd");
      DataOutBase::write_pvd_record (pvd_output, times_and_names);
    }
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::reconfigure_qc(const ConfigureQC &configure)
{
  configure_qc = configure;
  setup_cell_energy_molecules();
  setup_system();
  setup_fe_values_objects();
  update_neighbor_lists();
  update_positions();
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_triangulation()
{
  configure_qc.get_geometry<dim>()->create_mesh(triangulation);
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_cell_molecules()
{
  TimerOutput::Scope t (computing_timer, "Parse and assign all atoms to cells");

  if (!(configure_qc.get_atom_data_file()).empty() )
    {
      const std::string atom_data_file = configure_qc.get_atom_data_file();
      std::fstream fin(atom_data_file, std::fstream::in );
      cell_molecule_data =
        CellMoleculeTools::build_cell_molecule_data<dim>
        (fin,
         triangulation,
         configure_qc.get_ghost_cell_layer_thickness());
    }
  else if ( !(* configure_qc.get_stream()).eof() )
    cell_molecule_data =
      CellMoleculeTools::build_cell_molecule_data<dim>
      (*configure_qc.get_stream(),
       triangulation,
       configure_qc.get_ghost_cell_layer_thickness());
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

template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_cell_energy_molecules()
{
  TimerOutput::Scope t (computing_timer,
                        "Setup energy molecules with cluster weights");

  // It is ConfigureQC that actually creates a shared pointer to the derived
  // class object of the Cluster::WeightsByBase according to the parsed input.

  cluster_weights_method = configure_qc.get_cluster_weights<dim>();

  //TODO: Get Quadrature from ConfigureQC.
  cluster_weights_method->initialize (triangulation,
                                      QTrapez<dim>());

  cell_molecule_data.cell_energy_molecules =
    cluster_weights_method->
    update_cluster_weights (triangulation,
                            cell_molecule_data.cell_molecules);
}



template <int dim, typename PotentialType>
template<typename T>
void QC<dim, PotentialType>::write_mesh (T &os, const std::string &type )
{
  GridOut grid_out;
  if ( !type.compare("eps")  )
    grid_out.write_eps (triangulation, os);
  else if ( !type.compare("msh") )
    grid_out.write_msh (triangulation, os);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::initialize_boundary_functions()
{
  TimerOutput::Scope t (computing_timer, "Initialize boundary functions");

  std::map<unsigned int, std::vector<std::string> >
  boundary_ids_to_function_expressions = configure_qc.get_boundary_functions();

  for (auto &single_bc : boundary_ids_to_function_expressions)
    {
      const unsigned int n_components = single_bc.second.size();

      Assert (n_components == dim /* * atomicity*/,
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
                       std::make_pair
                       (component_mask,
                        std::make_shared<FunctionParser</*space*/dim>>(dim*1,
                            0.))
                      )
      );

      dirichlet_boundary_functions[single_bc.first].second->
      initialize (FunctionParser<dim>::default_variable_names(),
                  single_bc.second,
                  typename FunctionParser<dim>::ConstMap() /* TODO, true*/);
    }
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_boundary_conditions (const double)
{
  for (const auto &single_bc : dirichlet_boundary_functions)
    VectorTools::interpolate_boundary_values (dof_handler,
                                              single_bc.first,
                                              *(single_bc.second.second),
                                              constraints,
                                              single_bc.second.first);
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::initialize_external_potential_fields (const double initial_time)
{
  for (const auto &entry : configure_qc.get_external_potential_fields())
    {
      auto external_potential_field_iterator =
        external_potential_fields.insert
        (
          std::make_pair(entry.first.first,
                         std::make_shared<PotentialField<dim>>(entry.first.second,
                                                               initial_time))
        );

      // Initialize FunctionParser object of PotentialField.
      external_potential_field_iterator->second->
      initialize ((dim==3) ? "x,y,z,t" :
                  (dim==2  ? "x,y,t"   : "x,t"),
                  entry.second,
                  typename FunctionParser<dim>::ConstMap(),
                  true);
    }
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_system ()
{
  TimerOutput::Scope t (computing_timer, "Setup system");

  dof_handler.distribute_dofs (fe);

  locally_relevant_set =
    CellMoleculeTools::
    extract_locally_relevant_dofs (dof_handler,
                                   configure_qc.get_ghost_cell_layer_thickness());

  // set-up constraints objects
  constraints.reinit (locally_relevant_set);
  DoFTools::make_hanging_node_constraints (dof_handler, constraints);

  setup_boundary_conditions();

  constraints.close ();

  locally_relevant_gradient.reinit(dof_handler.locally_owned_dofs(),
                                   locally_relevant_set,
                                   mpi_communicator,
                                   true);
  locally_relevant_gradient = 0.;

  locally_relevant_displacement.reinit(dof_handler.locally_owned_dofs(),
                                       locally_relevant_set,
                                       mpi_communicator,
                                       false);

  locally_relevant_displacement = 0.;

  cells_to_data.clear();

  // TODO: use TriaAccessor<>::set_user_pointer() to associate AssemblyData with a cell
  for (types::DoFCellIteratorType<dim>
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    cells_to_data.insert (std::make_pair (cell,
                                          AssemblyData()));

  /*
  // FIXME: Do we want to initialize cell_to_data using locally relevant cells?
  //        Initializing it with all the cells is perhaps not necessary.
  // Initialize cells_to_data with all the cells in energy_atoms
  auto &energy_atoms = atom_data.energy_atoms;
  types::CellMoleculeConstIteratorType<dim> unique_key;
  for (unique_key  = energy_atoms.cbegin();
       unique_key != energy_atoms.cend();
       unique_key  = energy_atoms.upper_bound(unique_key->first))
    cells_to_data.insert(std::make_pair(unique_key->first, AssemblyData()));
   */

}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::setup_fe_values_objects ()
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

  // FIXME: Loop only over cells associated to energy molecules.
  for (types::DoFCellIteratorType<dim>
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    {
      // Include all the energy molecules associated to this active cell
      // as quadrature points. The quadrature points will be then used to
      // initialize fe_values object so as to evaluate the displacement
      // at the initial location of energy molecules in the active cells.

      const auto energy_molecules_range =
        CellMoleculeTools::molecules_range_in_cell<dim> (cell,
                                                         cell_energy_molecules);

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
      types::CellMoleculeIteratorType<dim>
      cell_energy_molecule_iterator =
        cell_energy_molecules.erase(energy_molecules_range.first.first,
                                    energy_molecules_range.first.first);

      for (unsigned int q = 0; q < n_energy_molecules_in_cell; ++q, ++cell_energy_molecule_iterator)
        {
          // const_iter->second yields the actual atom
          points[q]           = cell_energy_molecule_iterator->second.position_inside_reference_cell;
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
        std::make_shared<FEValues<dim>> (mapping, fe,
                                         Quadrature<dim> (points,
                                                          weights_per_atom),
                                         update_values);

      // finally reinit FEValues so that it's ready to provide all required
      // information:
      data.fe_values->reinit(cell);

      data.displacements.resize(points.size());
    }
}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::update_positions()
{
  TimerOutput::Scope t (computing_timer, "Update energy molecules' positions");

  // TODO: Loop over only locally relevant cells (ref FIXME in setup_system()).
  // First, loop over all cells and evaluate displacement field at quadrature
  // points. This is needed irrespectively of energy or gradient calculations.
  for (auto
       cell  = dof_handler.begin_active();
       cell != dof_handler.end();
       cell++)
    {
      const auto it = cells_to_data.find(cell);
      Assert (it != cells_to_data.end(),
              ExcInternalError());

      std::pair<types::CellMoleculeIteratorType<dim>, types::CellMoleculeIteratorType<dim> >
      cell_energy_molecules_range = cell_molecule_data.cell_energy_molecules.equal_range(cell);

      // FIXME: remove after FIXME in setup_system()
      // If this cell is not within the locally relevant active cells of the
      // current MPI process continue active cell loop
      if (cell_energy_molecules_range.first==cell_energy_molecules_range.second)
        continue;

      // get displacement field on all quadrature points of this object
      it->second.fe_values->operator[](u_fe).get_function_values(locally_relevant_displacement,
                                                                 it->second.displacements);
      const auto &displacements = it->second.displacements;

      // Update energy molecules positions.
      for (unsigned int i = 0;
           i < displacements.size();
           i++, ++cell_energy_molecules_range.first)
        // FIXME: loop over all atoms and use BlockVector for displacements
        cell_energy_molecules_range.first->second.atoms[0].position =
          cell_energy_molecules_range.first->second.atoms[0].initial_position +
          displacements[i];

      // The loop over displacements must have exhausted all the energy
      // molecules on a per cell basis (and the converse also should be true).
      Assert (cell_energy_molecules_range.first == cell_energy_molecules_range.second,
              ExcInternalError());
    }

}



template <int dim, typename PotentialType>
void QC<dim, PotentialType>::update_neighbor_lists()
{
  TimerOutput::Scope t (computing_timer, "Update neighbor lists");

  // TODO: Update neighbor lists
  // if( (iter_count % neigh_modify_delay)==0 || (max_abs_displacement > neigh_skin)   )
  //   update_neighbour_lists();
  neighbor_lists = molecule_handler.get_neighbor_lists(cell_molecule_data.cell_energy_molecules);
}



template <int dim, typename PotentialType>
template <bool ComputeGradient>
double QC<dim, PotentialType>::compute (vector_t &gradient) const
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



template <int dim, typename PotentialType>
template <bool ComputeGradient>
double QC<dim, PotentialType>::compute_local (vector_t &gradient) const
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
  const types::CellIteratorType<dim>
  cell_I_first = neighbor_lists.begin()->first.first,
  cell_J_first = neighbor_lists.begin()->first.second;

  // Convert tria's cells into dof cells.
  const types::DoFCellIteratorType<dim>
  dof_cell_I_first (&triangulation,
                    cell_I_first->level(),
                    cell_I_first->index(),
                    &dof_handler);

  const types::DoFCellIteratorType<dim>
  dof_cell_J_first (&triangulation,
                    cell_J_first->level(),
                    cell_J_first->index(),
                    &dof_handler);

  // Locate the first pair of cells I-J.
  typename
  std::map<types::DoFCellIteratorType<dim>, AssemblyData>::const_iterator
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
      const types::CellIteratorType<dim>
      &cell_I  = cell_pair_cell_molecule_pair.first.first,
       &cell_J = cell_pair_cell_molecule_pair.first.second;

      const types::CellMoleculeConstIteratorType<dim>
      &cell_molecule_I  = cell_pair_cell_molecule_pair.second.first,
       &cell_molecule_J = cell_pair_cell_molecule_pair.second.second;

      Assert ((cell_I == cell_molecule_I->first) &&
              (cell_J == cell_molecule_J->first),
              ExcMessage("Incorrect neighbor lists."
                         "Either cell_I or cell_J doesn't contain "
                         "cell_molecule_I or cell_molecule_J, respectively."));

      // FIXME: loop over all atoms
      const Tensor<1,dim> rIJ = cell_molecule_I->second.atoms[0].position -
                                cell_molecule_J->second.atoms[0].position;

      const double r_square = rIJ.norm_square();

      // If molecules I and J interact with each other while belonging to
      // different clusters. In this case, we need to account for
      // different weights associated with the clusters by
      // scaling E_{IJ} with (n_I + n_J)/2, which is exactly how
      // this contribution would be added had we followed assembly
      // from clusters perspective.
      // Since molecules not attributed to clusters have zero weights,
      // we can directly use the scaling above without the need to find out
      // whether or not one of the two molecules do not belong to any cluster.

      // Compute scaling of energy due to cluster weights.
      const double scale_energy = 0.5 * (cell_molecule_I->second.cluster_weight +
                                         cell_molecule_J->second.cluster_weight );

      // Assert that the scaling factor of energy, due to clustering, is
      // non-zero. One of the molecule must be a cluster molecule and
      // therefore have non-zero (more specifically positive) cluster_weight.
      Assert (scale_energy > 0, ExcInternalError());

      // Compute energy and gradient for a purely pair-wise interaction of
      // molecule I and  molecule J:
      const std::pair<double, double> pair =
        (*potential_ptr).template
        energy_and_gradient<ComputeGradient> (cell_molecule_I->second.atoms[0].type,
                                              cell_molecule_J->second.atoms[0].type,
                                              r_square);

      // Now we scale the energy according to cluster weights of the molecules.
      energy_per_process += scale_energy * pair.first ;

      if (ComputeGradient)
        {
          const types::DoFCellIteratorType<dim> dof_cell_I (&triangulation,
                                                            cell_I->level(),
                                                            cell_I->index(),
                                                            &dof_handler);
          const types::DoFCellIteratorType<dim> dof_cell_J (&triangulation,
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
          qI = cell_molecule_I->second.local_index,
          qJ = cell_molecule_J->second.local_index;

          const double r = std::sqrt(r_square);
          const double force_multiplier  = scale_energy * pair.second / r;

          // FIXME: evaluate gradients for all atoms in molecules.
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
              const unsigned int nonzero_comp = fe.system_to_component_index(k).first;

              local_gradient_I[k] += force_multiplier *
                                     rIJ[nonzero_comp] *
                                     cell_data_I->second.fe_values->shape_value(k, qI);

              local_gradient_J[k] -= force_multiplier *
                                     rIJ[nonzero_comp] *
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

template <int dim, typename PotentialType>
void QC<dim, PotentialType>::minimize_energy (const double time)
{
  if (time >= 0.)
    for (auto &potential_field : external_potential_fields)
      potential_field.second->set_time (time);

  // FIXME: move u and inv_mass to constructor
  vector_t u (QC<dim, PotentialType>::dof_handler.locally_owned_dofs(),
              QC<dim, PotentialType>::mpi_communicator);

  // Use this to initialize DiagonalMatrix
  u = 1.;

  // FIXME: Compute inverse mass matrix based on cluster weights.
  //        Make inverse matrix a member variable of this class.
  //        write a function to set it up.

  // Create inverse diagonal matrix.
  DiagonalMatrix<vector_t> inv_mass;
  inv_mass.reinit(u);
  //---------------------------------------------------------------------------

  std::function<double(vector_t &,  const vector_t &)> compute_function =
    [&]               (vector_t &G, const vector_t &U) -> double
  {
    locally_relevant_displacement = U;
    update_positions();
    const double energy = compute<true>(locally_relevant_gradient);
    G = locally_relevant_gradient;
    return energy;
  };

  u = 0.;

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
          solver.solve(compute_function, u, inv_mass);
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



/**
 * A macro that is used in instantiating QC class and it's functions
 * for 1d, 2d and 3d. Call this macro with the name of another macro that when
 * called with an integer argument and a PotentialType instantiates the
 * respective classes and functions in the given space dimension.
 */
#define DEAL_II_QC_INSTANTIATE(INSTANTIATIONS)      \
  INSTANTIATIONS(1, Potential::PairLJCutManager)    \
  INSTANTIATIONS(2, Potential::PairLJCutManager)    \
  INSTANTIATIONS(3, Potential::PairLJCutManager)    \
  INSTANTIATIONS(1, Potential::PairCoulWolfManager) \
  INSTANTIATIONS(2, Potential::PairCoulWolfManager) \
  INSTANTIATIONS(3, Potential::PairCoulWolfManager)

// Instantiations
#define INSTANTIATE(dim, PotentialType)                                      \
  template QC<dim, PotentialType>::QC (const ConfigureQC&);                  \
  template QC<dim, PotentialType>::~QC ();                                   \
  template void QC<dim, PotentialType>::reconfigure_qc (const ConfigureQC&); \
  template void QC<dim, PotentialType>::run ();                              \
  template void QC<dim, PotentialType>::setup_cell_molecules ();             \
  template void QC<dim, PotentialType>::setup_cell_energy_molecules ();      \
  template void QC<dim, PotentialType>::setup_boundary_conditions(const double); \
  template void QC<dim, PotentialType>::setup_system ();                     \
  template void QC<dim, PotentialType>::setup_triangulation();               \
  template void QC<dim, PotentialType>::write_mesh (std::ofstream &, const std::string &);        \
  template void QC<dim, PotentialType>::setup_fe_values_objects ();          \
  template void QC<dim, PotentialType>::update_positions();                  \
  template double QC<dim, PotentialType>::compute<true >(TrilinosWrappers::MPI::Vector &) const;  \
  template double QC<dim, PotentialType>::compute<false>(TrilinosWrappers::MPI::Vector &) const;  \
  template void QC<dim, PotentialType>::initialize_external_potential_fields (const double);      \
  template void QC<dim, PotentialType>::minimize_energy (const double);      \
  template void QC<dim, PotentialType>::output_results (const double, const unsigned int) const;

DEAL_II_QC_INSTANTIATE(INSTANTIATE)

DEAL_II_QC_NAMESPACE_CLOSE
