
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II-qc/atom/cell_atom_tools.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_lumped_vertex.h>

namespace dealiiqc
{

  namespace Cluster
  {



    template <int dim>
    WeightsByLumpedVertex<dim>::WeightsByLumpedVertex (const double &cluster_radius,
                                                       const double &maximum_cutoff_radius)
      :
      WeightsByBase<dim>(cluster_radius, maximum_cutoff_radius)
    {}



    template <int dim>
    types::CellAtomContainerType<dim>
    WeightsByLumpedVertex<dim>::update_cluster_weights (const types::MeshType<dim> &mesh,
                                                        const types::CellAtomContainerType<dim> &atoms) const
    {
      // Prepare energy atoms in this container.
      types::CellAtomContainerType<dim> energy_atoms;

      // Get the squared_energy_radius to identify energy atoms.
      const double squared_energy_radius =
        dealii::Utilities::fixed_power<2> (WeightsByBase<dim>::maximum_cutoff_radius +
                                           WeightsByBase<dim>::cluster_radius);

      // Get the squared_cluster_radius to identify cluster atoms.
      const double squared_cluster_radius =
        dealii::Utilities::fixed_power<2>(WeightsByBase<dim>::cluster_radius);

      // Get underlying p::Triangulation to construct DoFHandler
      const parallel::Triangulation<dim> *const ptria =
        dynamic_cast<const parallel::Triangulation<dim> *>
        (&mesh.get_triangulation());

      // Get a consistent MPI_Comm.
      const MPI_Comm &mpi_communicator = ptria != nullptr
                                         ?
                                         ptria->get_communicator()
                                         :
                                         MPI_COMM_SELF;

      // using linear mapping and linear scalar-valued FE
      MappingQ1<dim> mapping;
      FE_Q<dim> fe(1);

      DoFHandler<dim> dof_handler(*ptria);
      dof_handler.distribute_dofs(fe);

      // Get the total number of dofs, in the current case of using linear
      // scalar-valued FE this is exactly equal to the number of vertices.
      const unsigned int n_dofs = dof_handler.n_dofs();

      // Prepare b_I entries of b in this container.
      // Clusters are identified with the global dof indices.
      std::vector<double> b(n_dofs,0);

      // Prepare the A_II diagonal entries of A in this container.
      std::vector<double> A(n_dofs,0);

      // Container to store quadrature points and weights.
      std::vector<Point<dim>> points;
      std::vector<double> weights_per_atom;

      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      // Gather global indices of the local dofs here for a given cell.
      std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

      for (types::CellIteratorType<dim>
           cell  = dof_handler.begin_active();
           cell != dof_handler.end();
           cell++)
        {
          // Include all the atoms associated to this active cell as quadrature
          // points. The quadrature points will be then used to
          // initialize fe_values object so as to evaluate the shape function
          // values at the all the lattice sites in the atomistic system.

          // Get cell atoms range
          const auto cell_atoms_range =
            CellAtomTools::atoms_range_in_cell (cell, atoms);

          const types::CellAtomConstIteratorType<dim>
          &cell_atoms_range_begin = cell_atoms_range.first.first,
           &cell_atoms_range_end  = cell_atoms_range.first.second;

          // Prepare the total number of atoms in this cell here.
          // This is also the total number of quadrature points in this cell.
          const unsigned int n_atoms_in_current_cell = cell_atoms_range.second;

          // If this cell is not within the locally relevant active cells of the
          // current MPI process continue active cell loop
          if (n_atoms_in_current_cell == 0)
            continue;

          // Resize containers to known number of energy atoms in cell.
          points.resize(n_atoms_in_current_cell);
          weights_per_atom.resize(n_atoms_in_current_cell);

          types::CellAtomConstIteratorType<dim>
          cell_atom_iterator = cell_atoms_range_begin;
          for (unsigned int
               q = 0;
               q < n_atoms_in_current_cell;
               q++, cell_atom_iterator++)
            {
              Molecule<dim,1> molecule = cell_atom_iterator->second;
              const Point<dim> &position = molecule.atoms[0].position;

              // update quadrature point
              points[q] = molecule.position_inside_reference_cell;

              // Check the proximity of the atom to it's associated
              // cell's vertices.
              const auto closest_vertex =
                Utilities::find_closest_vertex (position,
                                                cell);

              if (closest_vertex.second < squared_energy_radius)
                {
                  if (closest_vertex.second < squared_cluster_radius)
                    {
                      // atom is cluster atom
                      molecule.cluster_weight = 1.;
                      weights_per_atom[q] = 1.;
                    }
                  else
                    {
                      // atom is not cluster atom
                      molecule.cluster_weight = 0.;
                      weights_per_atom[q] = 0.;
                    }

                  // Insert atom into energy_atoms if it is within a distance of
                  // energy_radius to associated cell's vertices.
                  energy_atoms.insert(std::make_pair(cell, molecule));
                }

            }

          Assert (cell_atom_iterator == cell_atoms_range_end,
                  ExcMessage("The number of energy atoms in the cell counted "
                             "using the distance between the iterator ranges "
                             "yields a different result than "
                             "incrementing the iterator to energy_atoms."
                             "Why wasn't this error thrown earlier?"));

          Assert (points.size() == weights_per_atom.size(),
                  ExcDimensionMismatch(points.size(), weights_per_atom.size()));

          // Do not need to compute b_I and A_I for ghost cells as we will
          // later sum contributions from all the processes.
          if (!cell->is_locally_owned())
            continue;

          // Now we are ready to initialize FEValues object for this cell.
          // Unfortunately, we need to set up a new FEValues object
          // as location of atoms associated to the cell is generally different
          // for each cell.
          FEValues<dim> fe_values (mapping,
                                   fe,
                                   Quadrature<dim>(points, weights_per_atom),
                                   update_values);
          fe_values.reinit (cell);

          cell->get_dof_indices(local_dofs);

          for (unsigned int i=0; i<local_dofs.size(); ++i)
            {
              const dealii::types::global_dof_index I = local_dofs[i];
              for (unsigned int q = 0; q < n_atoms_in_current_cell; q++)
                {
                  b[I] += fe_values.shape_value(i,q);
                  A[I] += fe_values.shape_value(i,q)*
                          fe_values.get_quadrature().weight(q);
                }
            }

        } // end of the loop over all active cells

      //---Finished adding energy atoms

      // Accumulate b entries per vertex from all MPI processes.
      dealii::Utilities::MPI::sum (b, mpi_communicator, b);

      // Accumulate A diagonal entries per vertex from all MPI processes.
      dealii::Utilities::MPI::sum (A, mpi_communicator, A);

      //---Finished updating b_per_cell and A_per_cell

      //---Now update cluster weights with correct value

      // Loop over all the energy atoms,
      // update their weights by multiplying with the factor
      // (b_per_cell/A_per_cell)
      for (auto &energy_atom : energy_atoms)
        {
          const auto &cell = energy_atom.first;
          Molecule<dim,1>  &molecule = energy_atom.second;

          // Get the closest vertex (of this cell) to the atom.
          const auto vertex_and_squared_distance =
            Utilities::find_closest_vertex (molecule.atoms[0].position,
                                            cell);

          // We need to get the global dof index from the local index of
          // the closest vertex.

          // we use scalar Q1 FEM, so we have one DoF per vertex,
          // thus as the second parameter to vertex_dof_index()
          // we provide zero.
          const dealii::types::global_dof_index I =
            cell->vertex_dof_index(vertex_and_squared_distance.first,0);

          Assert (A[I] != 0, ExcInternalError());

          // The cluster weight was previously set to 1. if the atom is
          // cluster atom and 0. if the atom is not cluster atom.
          energy_atom.second.cluster_weight *= b[I]
                                               /
                                               A[I];
        }

      return energy_atoms;
    }



    // Instantiations.
    template class WeightsByLumpedVertex<1>;
    template class WeightsByLumpedVertex<2>;
    template class WeightsByLumpedVertex<3>;


  } // namespace Cluster


} // namespace dealiiqc

