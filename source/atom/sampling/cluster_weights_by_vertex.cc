
#include <deal.II/base/utilities.h>

#include <deal.II-qc/atom/sampling/cluster_weights_by_vertex.h>

namespace dealiiqc
{

  namespace Cluster
  {



    template <int dim>
    WeightsByVertex<dim>::WeightsByVertex (const double &cluster_radius,
                                           const double &maximum_cutoff_radius)
      :
      WeightsByBase<dim>(cluster_radius, maximum_cutoff_radius)
    {}



    template <int dim>
    types::CellAtomContainerType<dim>
    WeightsByVertex<dim>::update_cluster_weights (const types::MeshType<dim> &mesh,
                                                  const types::CellAtomContainerType<dim> &atoms) const
    {
      // Prepare energy atoms in this container.
      types::CellAtomContainerType<dim> energy_atoms;

      const unsigned int n_vertices = mesh.get_triangulation().n_vertices();

      const parallel::Triangulation<dim> *const ptria =
        dynamic_cast<const parallel::Triangulation<dim> *>
        (&mesh.get_triangulation());

      const MPI_Comm &mpi_communicator = ptria != nullptr
                                         ?
                                         ptria->get_communicator()
                                         :
                                         MPI_COMM_SELF;

      // Prepare the total number of atoms per vertex in this container.
      // The container should also contain the information of the total number
      // of atoms per per vertex for ghost cells on the current MPI process.
      std::vector<unsigned int> n_atoms_per_vertex(n_vertices,0);

      // Prepare the number of cluster atoms per vertex in this container.
      std::vector<unsigned int> n_cluster_atoms_per_vertex(n_vertices,0);

      // Get the squared_energy_radius to identify energy atoms.
      const double squared_energy_radius =
        dealii::Utilities::fixed_power<2> (WeightsByBase<dim>::maximum_cutoff_radius +
                                           WeightsByBase<dim>::cluster_radius);

      // Get the squared_cluster_radius to identify cluster atoms.
      const double squared_cluster_radius =
        dealii::Utilities::fixed_power<2>(WeightsByBase<dim>::cluster_radius);

      // Loop over all atoms, see if a given atom is energy atom and
      // if so if it's a cluster atom.
      // While there, count the total number of atoms per vertex and
      // number of cluster atoms per vertex.
      for (const auto &cell_atom : atoms)
        {
          const auto &cell = cell_atom.first;
          Molecule<dim,1> molecule   = cell_atom.second;

          // Get the closest vertex (of this cell) to the atom.
          const auto vertex_and_squared_distance =
            Utilities::find_closest_vertex (molecule.initial_position,
                                            cell);

          const unsigned int global_vertex_index =
            cell->vertex_index(vertex_and_squared_distance.first);

          const double &squared_distance_from_closest_vertex =
            vertex_and_squared_distance.second;

          // Count only atoms assigned to locally owned cells, so that
          // when sum is performed over all MPI processes only the atoms
          // assigned locally owned cells are counted.
          if (cell->is_locally_owned())
            n_atoms_per_vertex[global_vertex_index]++;

          if (squared_distance_from_closest_vertex < squared_energy_radius)
            {
              if (squared_distance_from_closest_vertex < squared_cluster_radius)
                {
                  // Count only the cluster atoms of the locally owned cells.
                  if (cell->is_locally_owned())
                    // Increment cluster atom count for this "vertex"
                    n_cluster_atoms_per_vertex[global_vertex_index]++;
                  // atom is cluster atom
                  molecule.cluster_weight = 1.;
                }
              else
                // atom is not cluster atom
                molecule.cluster_weight = 0.;

              // Insert atom into energy_atoms if it is within a distance of
              // energy_radius to associated cell's vertices.
              energy_atoms.insert(std::make_pair(cell,molecule));
            }
        }

      //---Finished adding energy atoms

      // Accumulate the number of atoms per vertex from all MPI processes.
      dealii::Utilities::MPI::sum (n_atoms_per_vertex,
                                   mpi_communicator,
                                   n_atoms_per_vertex);

      // Accumulate the number of cluster atoms per vertex from all MPI processes.
      dealii::Utilities::MPI::sum (n_cluster_atoms_per_vertex,
                                   mpi_communicator,
                                   n_cluster_atoms_per_vertex);

      //---Now update cluster weights with correct value

      // Loop over all the energy atoms,
      // update their weights by multiplying with the factor
      // (n_atoms/n_cluster_atoms)
      for (auto &energy_atom : energy_atoms)
        {
          const auto &cell = energy_atom.first;
          Molecule<dim,1>  &molecule = energy_atom.second;

          // Get the closest vertex (of this cell) to the atom.
          const auto vertex_and_squared_distance =
            Utilities::find_closest_vertex (molecule.initial_position,
                                            cell);

          const unsigned int global_vertex_index =
            cell->vertex_index(vertex_and_squared_distance.first);

          Assert (n_cluster_atoms_per_vertex[global_vertex_index] != 0,
                  ExcInternalError());

          // The cluster weight was previously set to 1. if the atom is
          // cluster atom and 0. if the atom is not cluster atom.
          molecule.cluster_weight *=
            static_cast<double>(n_atoms_per_vertex[global_vertex_index])
            /
            static_cast<double>(n_cluster_atoms_per_vertex[global_vertex_index]);
        }

      return energy_atoms;
    }



    // Instantiations.
    template class WeightsByVertex<1>;
    template class WeightsByVertex<2>;
    template class WeightsByVertex<3>;


  } // namespace Cluster


} // namespace dealiiqc

