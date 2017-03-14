// a source file which contains definition of core functions of QC class
#include <dealiiqc/qc.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  QC<dim>::~QC ()
  {
    dof_handler.clear();
  }

  template <int dim>
  QC<dim>::QC ( const ConfigureQC &config )
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
    dof_handler    (triangulation),
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

    setup_system();
    associate_atoms_with_cells();

  }

  template <int dim>
  void QC<dim>::setup_triangulation()
  {
    if (!(configure_qc.get_mesh_file()).empty() )
      {
        const std::string meshfile = configure_qc.get_mesh_file();
        GridIn<dim> gridin;
        gridin.attach_triangulation( triangulation );
        std::ifstream fin( meshfile );
        gridin.read_msh(fin);
      }
    else
      {
        GridGenerator::hyper_cube (triangulation);
      }
    if ( configure_qc.get_n_initial_global_refinements() )
      triangulation.refine_global(configure_qc.get_n_initial_global_refinements());
  }

  template <int dim>
  void QC<dim>::setup_atoms()
  {
    if (!(configure_qc.get_atom_data_file()).empty() )
      {
        const std::string atom_data_file = configure_qc.get_atom_data_file();
        std::stringstream ss;
        std::fstream fin(atom_data_file, std::fstream::in );
        ss << fin.rdbuf();
        fin.close();
        ParseAtomData<dim> atom_parser;
        // TODO: Use atom types to initialize neighbor lists faster
        // TODO: Use masses of different types of atom for FIRE minimization scheme?
        std::map<unsigned int, types::global_atom_index> atom_types;
        std::vector<double> masses;
        atom_parser.parse( ss, atoms, masses, atom_types);
      }
    else if ( !(* configure_qc.get_stream()).eof() )
      {
        ParseAtomData<dim> atom_parser;
        // TODO: Use atom types to initialize neighbor lists faster
        // TODO: Use masses of different types of atom for FIRE minimization scheme?
        std::map<unsigned int, types::global_atom_index> atom_types;
        std::vector<double> masses;
        atom_parser.parse( *configure_qc.get_stream(), atoms, masses, atom_types);
      }
    else
      Assert(false, ExcMessage("None of the atom attributes set!"));
  }

  template <int dim>
  template<typename T>
  void QC<dim>::write_mesh( T &os, const std::string &type )
  {
    GridOut grid_out;
    if ( !type.compare("eps")  )
      grid_out.write_eps (triangulation, os);
    else if ( !type.compare("msh") )
      grid_out.write_msh (triangulation, os);
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  void QC<dim>::setup_system ()
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
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      cells_to_data.insert(std::make_pair(cell,AssemblyData()));
  }

  template <int dim>
  void QC<dim>::setup_fe_values_objects ()
  {
    // vector of atoms we care about for calculation, i.e. those within
    // the clusters plus those in the cut-off:
    std::vector<Point<dim>> points;
    std::vector<double> weights_per_atom;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      {
        // vector of atoms we care about for calculation, i.e. those within
        // the clusters plus those in the cut-off:
        points.resize(0);
        weights_per_atom.resize(0);

        AssemblyData &data = cells_to_data[cell];

        // for now take all points as relevant:
        for (unsigned int q = 0; q < data.cell_atoms.size(); q++)
          {
            const unsigned int aid = data.cell_atoms[q];
            points.push_back(atoms[aid].reference_position);
            weights_per_atom.push_back(1.0); // TODO: put cluster weights here and 0. for those outside of clusters.
            data.quadrature_atoms[aid] = q;
            data.energy_atoms.push_back(aid);
          }

        Assert (points.size() == weights_per_atom.size(),
                ExcDimensionMismatch(points.size(), weights_per_atom.size()));

        Assert (points.size() > 0,
                ExcMessage("Cell does not have any atoms at which fields and "
                           "shape functions are to be evaluated."));

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

  template <int dim>
  double QC<dim>::calculate_energy_gradient(const vector_t &locally_relevant_displacement,
                                            vector_t &gradient) const
  {
    double res = 0.;
    gradient = 0.;

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
      }

    // TODO: Update neighbour lists
    // if( (iter_count % neigh_modify_delay)==0 || (max_abs_displacement > neigh_skin)   )
    //   update_neighbour_lists();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double> local_gradient(dofs_per_cell);


    // TODO: parallelize using multithreading by dropping i>j and doing 1/2 ?
    // We can't really enforce that two threads won't try to simultaneously
    // calculate E_{ij} and E_{ji} where i and j belong to neighboring cells.
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      {
        local_gradient = 0.;
        const auto it = cells_to_data.find(cell);
        Assert (it != cells_to_data.end(),
                ExcInternalError());

        cell->get_dof_indices (local_dof_indices);

        // for each cell, go trhough all atoms we care about in energy calculation
        for (unsigned int a = 0; a < it->second.energy_atoms.size(); a++)
          {
            // global id of current atom:
            const unsigned int  I = it->second.energy_atoms[a];
            const auto qI_it = it->second.quadrature_atoms.find(I);
            const unsigned int qI = qI_it->second;

            // Current position of atom:
            const Point<dim> xI = atoms[I].position + it->second.displacements[qI];

            // loop over all neighbours and disregard J<I
            // TODO: implement ^^^^
            // for now there is always one neighbour only: I+1
            if (I+1 < atoms.size())
              {
                const unsigned int J = I+1;
                // get Data for neighbour atom
                // TODO: check if the atom is in this cell also to save some time.
                const auto n_data = cells_to_data.find(atoms[J].parent_cell);

                // now we need to know what is the quadrature point number
                // associated with the atom J
                const auto qJ_it = n_data->second.quadrature_atoms.find(J);
                Assert (qJ_it != n_data->second.quadrature_atoms.end(),
                        ExcInternalError());
                const unsigned int qJ = qJ_it->second;

                // current position of atom J
                const Point<dim> xJ = atoms[J].position + n_data->second.displacements[qJ];

                // distance vector
                const Tensor<1,dim> rIJ = xI - xJ;

                const double r = rIJ.norm();

                // If atoms I and J interact with each other while belonging
                // different clusters. In this case, we need to account for
                // different weights associated with the clusters by
                // scaling E_{IJ} with (n_I + n_J)/2, which is exactly how
                // this contribution would be added had we followed assembly
                // from clusters perspective.
                // Here we need to distinguish between two cases: both atoms
                // belong to clusters (weight is as above), or
                // only the main atom belongs to a claster (weight is n_I/2)


                // Now we can calculate energy:
                // TODO: generalized, energy depends on a 2-points potential
                // used for atoms I and J. Could be different for any combination
                // of atoms.
                const double energy = 0.5 * Utilities::fixed_power<2>(r - 0.25);
                const double deriv  = r - 0.25;

                // Finally, we evaluated local contribution to the gradient of
                // energy. Here we need to distinguish between two cases:
                // 1. N_k(X_j) is non-zero on (possibly) neihboring cell
                // 2. N_k(X_j) is zero, i.e. X_j does not belong to the support
                // of N_k.
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    const auto k_neigh = n_data->second.global_to_local_dof.find(local_dof_indices[k]);
                    if (k_neigh == n_data->second.global_to_local_dof.end())
                      {
                        local_gradient[k] += (deriv / r) * rIJ *
                                             it->second.fe_values->operator[](u_fe).value(k, qI);
                      }
                    else
                      {
                        local_gradient[k] += (deriv / r) * rIJ *
                                             (it->second.fe_values->operator[](u_fe).value(k, qI) -
                                              n_data->second.fe_values->operator[](u_fe).value(k_neigh->second, qJ));
                      }
                  }

                res += energy;
              } // end of the loop over all neighbours

          } // end of the loop over all atoms

        // distribute gradient to the RHS:
        constraints.distribute_local_to_global(local_gradient,
                                               local_dof_indices,
                                               gradient);

      } // end of the loop over all cells

    return res;
  }


  template <int dim>
  void QC<dim>::run ()
  {
    setup_fe_values_objects();

    const double e = calculate_energy_gradient(locally_relevant_displacement,
                                               gradient);
  }


  template <int dim>
  void QC<dim>::associate_atoms_with_cells ()
  {
    TimerOutput::Scope t (computing_timer, "Associate atoms with cells");

    for (unsigned int i = 0; i < atoms.size(); i++)
      {
        Atom<dim> &a = atoms[i];
        const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>
            my_pair = GridTools::find_active_cell_around_point(mapping, dof_handler, a.position);

        a.reference_position = GeometryInfo<dim>::project_to_unit_cell(my_pair.second);
        a.parent_cell = my_pair.first;

        // add this atom to cell
        auto data = cells_to_data.find(a.parent_cell);
        Assert (data != cells_to_data.end(),
                ExcInternalError());
        data->second.cell_atoms.push_back(i);
      }
    // Check if all atoms are associated
    size_t atom_count =0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); cell++)
      atom_count += cells_to_data[cell].cell_atoms.size();

    Assert( atom_count==atoms.size(), ExcInternalError() );

  }

  // instantiations:
  // TODO: move to insta.in
  template void QC<1>::run ();
  template void QC<2>::run ();
  template void QC<3>::run ();
  template void QC<1>::setup_system ();
  template void QC<2>::setup_system ();
  template void QC<3>::setup_system ();
  template QC<1>::~QC ();
  template QC<2>::~QC ();
  template QC<3>::~QC ();
  template QC<1>::QC (const ConfigureQC &);
  template QC<2>::QC (const ConfigureQC &);
  template QC<3>::QC (const ConfigureQC &);
  template void QC<1>::setup_triangulation();
  template void QC<2>::setup_triangulation();
  template void QC<3>::setup_triangulation();
  template void QC<1>::write_mesh<std::ofstream>( std::ofstream &, const std::string &);
  template void QC<2>::write_mesh<std::ofstream>( std::ofstream &, const std::string &);
  template void QC<3>::write_mesh<std::ofstream>( std::ofstream &, const std::string &);
  template void QC<1>::associate_atoms_with_cells ();
  template void QC<2>::associate_atoms_with_cells ();
  template void QC<3>::associate_atoms_with_cells ();
  template void QC<1>::setup_fe_values_objects ();
  template void QC<2>::setup_fe_values_objects ();
  template void QC<3>::setup_fe_values_objects ();
  template double QC<1>::calculate_energy_gradient(TrilinosWrappers::MPI::Vector const &, TrilinosWrappers::MPI::Vector &) const;

}
