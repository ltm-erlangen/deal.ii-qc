// a source file which contains definition of core functions of QC class
#include <dealiiqc/qc.h>
#include <deal.II/grid/grid_tools.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  QC<dim>::~QC ()
  {
    dof_handler.clear();
  }

  template <int dim>
  QC<dim>::QC (/*const Parameters<dim> &parameters*/)
    :
    mpi_communicator(MPI_COMM_WORLD),
    pcout (std::cout,
           (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
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
    // TODO: read from input file
    const unsigned int N = 4;
    atoms.resize(N+1);
    const double L = 1.;
    for (unsigned int i = 0; i <= N; i++)
      {
        Point<dim> p;
        p[0] = (L*i)/N;
        atoms[i].position = p;
      }
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
  }

  template <int dim>
  void QC<dim>::setup_fe_values_objects ()
  {
    // vector of atoms we care about for calculation, i.e. those within
    // the clusters plus those in the cut-off:
    std::vector<Point<dim>> points;
    std::vector<double> weights_per_atom;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

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

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
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

                // shape function of l-th DoF evaluated at J-th atom:
                const unsigned int l = 0;
                const Tensor<1,dim> shape_k = n_data->second.fe_values->operator[](u_fe).value(l, qJ);

                // current position of atom J
                const Point<dim> xJ = atoms[J].position + n_data->second.displacements[qJ];

                // distance vector
                const Tensor<1,dim> rIJ = xI - xJ;

                const double r = rIJ.norm();

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
    // TODO: read .gmsh file
    {
      GridGenerator::hyper_cube (triangulation);
      triangulation.refine_global(1);
    }

    setup_system();
    associate_atoms_with_cells();
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
        cells_to_data[a.parent_cell].cell_atoms.push_back(i);
      }
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
  template QC<1>::QC ();
  template QC<2>::QC ();
  template QC<3>::QC ();
  template void QC<1>::associate_atoms_with_cells ();
  template void QC<2>::associate_atoms_with_cells ();
  template void QC<3>::associate_atoms_with_cells ();
  template void QC<1>::setup_fe_values_objects ();
  template void QC<2>::setup_fe_values_objects ();
  template void QC<3>::setup_fe_values_objects ();
  template double QC<1>::calculate_energy_gradient(TrilinosWrappers::MPI::Vector const&, TrilinosWrappers::MPI::Vector&) const;

}
