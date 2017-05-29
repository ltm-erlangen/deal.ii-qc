
// a source file which contains definition of core functions of QC class

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II-qc/atom/cell_atom_tools.h>
#include <deal.II-qc/core/qc.h>

namespace dealiiqc
{
  using namespace dealii;



  template <int dim, typename PotentialType>
  QC<dim, PotentialType>::~QC ()
  {
    dof_handler.clear();
  }



  template <int dim, typename PotentialType>
  QC<dim, PotentialType>::QC ( const ConfigureQC &config )
    :
    mpi_communicator(MPI_COMM_WORLD),
    pcout (std::cout,
           (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    configure_qc( config ),
    triangulation (mpi_communicator,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    fe (FE_Q<dim>(1),dim),
    u_fe (0),
    dof_handler (triangulation),
    atom_handler (configure_qc),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::never,
                     TimerOutput::wall_times)
  {
    Assert( dim==configure_qc.get_dimension(), ExcInternalError());

    // Load the mesh by reading from mesh file
    setup_triangulation();

    // Read atom data file and initialize atoms
    setup_atoms();
  }



  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::run ()
  {
    setup_energy_atoms_with_cluster_weights();
    setup_system();
    setup_fe_values_objects();
    update_neighbor_lists();
    update_energy_atoms_positions();
    const double e = calculate_energy_gradient(gradient);
    (void)e;
  }


  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::reconfigure_qc(const ConfigureQC &configure)
  {
    configure_qc = configure;
    setup_energy_atoms_with_cluster_weights();
    setup_system();
    setup_fe_values_objects();
    update_neighbor_lists();
    update_energy_atoms_positions();
  }


  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::setup_triangulation()
  {
    configure_qc.get_geometry<dim>()->create_mesh(triangulation);
  }



  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::setup_atoms()
  {
    TimerOutput::Scope t (computing_timer, "Parse and assign all atoms to cells");

    atom_handler.parse_atoms_and_assign_to_cells (dof_handler, atom_data);

    // It is ConfigureQC that actually creates a PotentialType object according
    // to the parsed input and can return a shared pointer to the PotentialType
    // object. However, charges in PotentialType object aren't set yet.
    // Finish setting up PotentialType object here.
    configure_qc.get_potential()->set_charges(atom_data.charges);
  }

  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::setup_energy_atoms_with_cluster_weights()
  {
    TimerOutput::Scope t (computing_timer, "Setup energy atoms with cluster weights");

    // It is ConfigureQC that actually creates a shared pointer to the derived
    // class object of the Cluster::WeightsByBase according to the parsed input.
    atom_data.energy_atoms =
      configure_qc.get_cluster_weights<dim>()->
      update_cluster_weights (dof_handler,
                              atom_data.atoms);
  }



  template <int dim, typename PotentialType>
  template<typename T>
  void QC<dim, PotentialType>::write_mesh( T &os, const std::string &type )
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
  void QC<dim, PotentialType>::setup_system ()
  {
    TimerOutput::Scope t (computing_timer, "Setup system");

    dof_handler.distribute_dofs (fe);

    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_set);

    // set-up constraints objects
    constraints.reinit (locally_relevant_set);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);

    /*
    std::set<types::boundary_id>       dirichlet_boundary_ids;
    typename FunctionMap<dim>::type    dirichlet_boundary_functions;
    ZeroFunction<dim>                  homogeneous_dirichlet_bc (1);
    dirichlet_boundary_ids.insert(0);
    dirichlet_boundary_functions[0] = &homogeneous_dirichlet_bc;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    */
    constraints.close ();

    displacement.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
    locally_relevant_displacement.reinit(locally_relevant_set, mpi_communicator);

    displacement = 0.;
    locally_relevant_displacement = displacement;

    cells_to_data.clear();

    // TODO: use TriaAccessor<>::set_user_pointer() to associate AssemblyData with a cell
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      cells_to_data.insert(std::make_pair(cell,AssemblyData()));

    /*
    // FIXME: Do we want to initialize cell_to_data using cells in energy_atoms?
    //        Initializing it with all the cells is perhaps not necessary.
    // Initialize cells_to_data with all the cells in energy_atoms
    auto &energy_atoms = atom_data.energy_atoms;
    types::CellAtomConstIteratorType<dim> unique_key;
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

    // Get a non-constant reference to energy atoms to update
    // local index (within a cell) of energy atoms.
    auto &energy_atoms = atom_data.energy_atoms;

    // FIXME: Loop only over cells in energy_atoms
    for (types::CellIteratorType<dim> cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
      {
        // Include all the energy_atoms associated to this active cell
        // as quadrature points. The quadrature points will be then used to
        // initialize fe_values object so as to evaluate the displacement
        // at the location of energy_atoms in the active cell.

        const auto energy_atoms_range =
          CellAtomTools::atoms_range_in_cell (cell,
                                              energy_atoms);

        const unsigned int n_energy_atoms_in_cell = energy_atoms_range.second;

        // If this cell is not within the locally relevant active cells of the
        // current MPI process continue active cell loop
        if (n_energy_atoms_in_cell == 0)
          continue;

        // Resize containers to known number of energy atoms in cell.
        points.resize(n_energy_atoms_in_cell);
        weights_per_atom.resize(n_energy_atoms_in_cell);

        AssemblyData &data = cells_to_data[cell];

        // We need non-const iterator to update local index of energy atom.
        // TODO: Move the task of updating local index of energy atoms to
        //       somewhere else? For now keeping it here.
        // To get a non-const iterator to the beginning of energy atom range
        // call erase with the same argument. Strictly no erase is performed
        // but erase would return a non-const iterator.
        // This is not exactly a hack as we are using how erase on containers
        // must behave. Also, this is a constant time operation.
        types::CellAtomIteratorType<dim>
        cell_atom_iterator = energy_atoms.erase(energy_atoms_range.first.first,
                                                energy_atoms_range.first.first);

        for (unsigned int q = 0; q < n_energy_atoms_in_cell; ++q, ++cell_atom_iterator)
          {
            // const_iter->second yields the actual atom
            points[q]           = cell_atom_iterator->second.reference_position;
            weights_per_atom[q] = cell_atom_iterator->second.cluster_weight;
            cell_atom_iterator->second.local_index = q;
          }

        Assert (cell_atom_iterator == energy_atoms_range.first.second,
                ExcMessage("The number of energy atoms in the cell counted "
                           "using the distance between the iterator ranges "
                           "yields a different result than "
                           "incrementing the iterator to energy_atoms."
                           "Why wasn't this error thrown earlier?"));

        Assert (points.size() == weights_per_atom.size(),
                ExcDimensionMismatch(points.size(), weights_per_atom.size()));

        Assert (data.fe_values.use_count() ==0,
                ExcInternalError());

        // Now we are ready to initialize FEValues object.
        data.fe_values = std::make_shared<FEValues<dim>>(mapping, fe,
                                                         Quadrature<dim>(points, weights_per_atom),
                                                         update_values);

        // finally reinit FEValues so that it's ready to provide all required
        // information:
        data.fe_values->reinit(cell);

        data.displacements.resize(points.size());

        // store global DoF -> local DoF map:
        cell->get_dof_indices(local_dof_indices);

        data.global_to_local_dof.clear();
        for (unsigned int i = 0; i < local_dof_indices.size(); i++)
          data.global_to_local_dof[local_dof_indices[i]] = i;
      }
  }



  template <int dim, typename PotentialType>
  void QC<dim, PotentialType>::update_energy_atoms_positions()
  {
    TimerOutput::Scope t (computing_timer, "Update energy atoms' positions");

    // First, loop over all cells and evaluate displacement field at quadrature
    // points. This is needed irrespectively of energy or gradient calculations.
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      {
        const auto it = cells_to_data.find(cell);
        Assert (it != cells_to_data.end(),
                ExcInternalError());

        // get displacement field on all quadrature points of this object
        it->second.fe_values->operator[](u_fe).get_function_values(locally_relevant_displacement,
                                                                   it->second.displacements);
        const auto &displacements = it->second.displacements;

        std::pair<types::CellAtomIteratorType<dim>, types::CellAtomIteratorType<dim> >
        cell_atom_range = atom_data.energy_atoms.equal_range(cell);

        // update energy atoms positions
        // TODO: write test to check if positions are updated correctly.
        for (unsigned int i = 0;
             i < displacements.size();
             i++, ++cell_atom_range.first)
          cell_atom_range.first->second.position += displacements[i];

        // The loop over displacements must have exhausted all the energy_atoms
        // on a per cell basis (and the converse also should be true).
        Assert (cell_atom_range.first==cell_atom_range.second,
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
    neighbor_lists = atom_handler.get_neighbor_lists(atom_data.energy_atoms);
  }



  template <int dim, typename PotentialType>
  template <bool ComputeGradient>
  double QC<dim, PotentialType>::calculate_energy_gradient (vector_t &gradient) const
  {
    TimerOutput::Scope t (computing_timer, "Compute energy and gradient");

    double energy_per_process = 0.;

    if (ComputeGradient)
      gradient = 0.;

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

    typename
    std::map<types::CellIteratorType<dim>, AssemblyData>::const_iterator
    cell_data_I = cells_to_data.begin(),
    cell_data_J = cells_to_data.begin();

    // TODO: parallelize using using TBB by looping over pairs of cells.
    // Rework neighbor list as std::map<std::pair<Cell,Cell>,....>
    for (const auto &cell_pair_cell_atom_pair : neighbor_lists)
      {
        // get reference to current cell pair and atom pair
        const types::CellIteratorType<dim>
        &cell_I  = cell_pair_cell_atom_pair.first.first,
         &cell_J = cell_pair_cell_atom_pair.first.second;

        const types::CellAtomConstIteratorType<dim>
        &cell_atom_I  = cell_pair_cell_atom_pair.second.first,
         &cell_atom_J = cell_pair_cell_atom_pair.second.second;

        Assert ((cell_I == cell_atom_I->first) &&
                (cell_J == cell_atom_J->first),
                ExcMessage("Incorrect neighbor lists."
                           "Either cell_I or cell_J doesn't contain "
                           "cell_atom_I or cell_atom_J, respectively."));

        const Tensor<1,dim> rIJ = cell_atom_I->second.position -
                                  cell_atom_J->second.position;

        const double r_square = rIJ.norm_square();

        // If atoms I and J interact with each other while belonging to
        // different clusters. In this case, we need to account for
        // different weights associated with the clusters by
        // scaling E_{IJ} with (n_I + n_J)/2, which is exactly how
        // this contribution would be added had we followed assembly
        // from clusters perspective.
        // Since atoms not attributed to clusters have zero weights,
        // we can directly use the scaling above without the need to find out
        // whether or not one of the two atoms do not belong to any cluster.

        // Compute scaling of energy due to cluster weights.
        const double scale_energy = 0.5 * (cell_atom_I->second.cluster_weight +
                                           cell_atom_J->second.cluster_weight );

        // Assert that the scaling factor of energy, due to clustering, is
        // non-zero. One of the atom must be a cluster atom and
        // therefore have non-zero (more specifically positive) cluster_weight.
        Assert (scale_energy > 0, ExcInternalError());

        // Compute energy and gradient for a purely pair-wise interaction of
        // atom I and  atom J:
        const std::pair<double, double> pair =
          (*potential_ptr).template
          energy_and_gradient<ComputeGradient> (cell_atom_I->second.type,
                                                cell_atom_J->second.type,
                                                r_square);

        // Now we scale the energy according to cluster weights of the atoms.
        energy_per_process += scale_energy * pair.first ;

        if (ComputeGradient)
          {
            // Check if I'th or Jth cell changed, if so, update the pointer to
            // the cell data and get dof indices.
            if (cell_data_I->first != cell_I)
              {
                cell_data_I = cells_to_data.find(cell_I);
                cell_I->get_dof_indices (local_dof_indices_I);
                local_gradient_I = 0.;
              }
            if (cell_data_J->first != cell_J)
              {
                cell_data_J = cells_to_data.find(cell_J);
                cell_J->get_dof_indices (local_dof_indices_J);
                local_gradient_J = 0.;
              }

            // TODO: Write test for update_energy_atoms_positions()
            // TODO: Complete gradient computation.
            const double deriv  = scale_energy * pair.second;

            // Get quadrature point index from the local_index of atom pairs
            const unsigned int
            qI = cell_atom_I->second.local_index,
            qJ = cell_atom_J->second.local_index;

            const double r = std::sqrt(r_square);

            // Finally, we evaluated local contribution to the gradient of
            // energy. Here we need to distinguish between two cases:
            // 1. N_k(X_j) is non-zero on (possibly) neighboring cell
            // 2. N_k(X_j) is zero, i.e. X_j does not belong to the support
            // of N_k.
            // Here k is the index of local shape function on the cell
            // where atom I belongs.
            // TODO: fix missing evaluation of forces on DoFs at cell J
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                const auto k_neigh = cell_data_J->second.global_to_local_dof.find(local_dof_indices_I[k]);
                if (k_neigh == cell_data_J->second.global_to_local_dof.end())
                  {
                    local_gradient_I[k] += (deriv / r) * rIJ *
                                           cell_data_I->second.fe_values->operator[](u_fe).value(k, qI);
                  }
                else
                  {
                    local_gradient_I[k] += (deriv / r) * rIJ *
                                           (cell_data_I->second.fe_values->operator[](u_fe).value(k, qI) -
                                            cell_data_J->second.fe_values->operator[](u_fe).value(k_neigh->second, qJ));
                  }
              }

            // FIXME: distribute local gradients to the RHS ONLY when cells change!!
            constraints.distribute_local_to_global(local_gradient_I,
                                                   local_dof_indices_I,
                                                   gradient);
            constraints.distribute_local_to_global(local_gradient_J,
                                                   local_dof_indices_J,
                                                   gradient);
          }

      }

    // sum contributions from all MPI cores and return the result:
    return dealii::Utilities::MPI::sum(energy_per_process, mpi_communicator);
  }



  /**
   * A macro that is used in instantiating QC class and it's functions
   * for 1d, 2d and 3d. Call this macro with the name of another macro that when
   * called with an integer argument and a PotentialType instantiates the
   * respective classes and functions in the given space dimension.
   */
#define DEAL_II_QC_INSTANTIATE(INSTANTIATIONS) \
  INSTANTIATIONS(1, Potential::PairLJCutManager) \
  INSTANTIATIONS(2, Potential::PairLJCutManager) \
  INSTANTIATIONS(3, Potential::PairLJCutManager) \
  INSTANTIATIONS(1, Potential::PairCoulWolfManager) \
  INSTANTIATIONS(2, Potential::PairCoulWolfManager) \
  INSTANTIATIONS(3, Potential::PairCoulWolfManager)

// Instantiations
#define INSTANTIATE(dim, PotentialType) \
  template QC<dim, PotentialType>::QC (const ConfigureQC&); \
  template QC<dim, PotentialType>::~QC (); \
  template void QC<dim, PotentialType>::reconfigure_qc (const ConfigureQC&); \
  template void QC<dim, PotentialType>::run (); \
  template void QC<dim, PotentialType>::setup_atoms (); \
  template void QC<dim, PotentialType>::setup_energy_atoms_with_cluster_weights (); \
  template void QC<dim, PotentialType>::setup_system (); \
  template void QC<dim, PotentialType>::setup_triangulation(); \
  template void QC<dim, PotentialType>::write_mesh<std::ofstream> (std::ofstream &, const std::string &); \
  template void QC<dim, PotentialType>::setup_fe_values_objects (); \
  template void QC<dim, PotentialType>::update_energy_atoms_positions(); \
  template double QC<dim, PotentialType>::calculate_energy_gradient<true >(TrilinosWrappers::MPI::Vector &) const; \
  template double QC<dim, PotentialType>::calculate_energy_gradient<false>(TrilinosWrappers::MPI::Vector &) const;

  DEAL_II_QC_INSTANTIATE(INSTANTIATE)

} // namespace dealiiqc
