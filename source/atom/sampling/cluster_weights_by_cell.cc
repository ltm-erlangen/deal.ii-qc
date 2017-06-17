
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>

namespace dealiiqc
{

  namespace Cluster
  {



    template <int dim>
    WeightsByCell<dim>::WeightsByCell (const double &cluster_radius,
                                       const double &maximum_cutoff_radius)
      :
      WeightsByBase<dim>(cluster_radius, maximum_cutoff_radius)
    {}



    template <int dim>
    types::CellAtomContainerType<dim>
    WeightsByCell<dim>::update_cluster_weights (const types::MeshType<dim> &mesh,
                                                const types::CellAtomContainerType<dim> &atoms) const
    {
      // Prepare energy atoms in this container.
      types::CellAtomContainerType<dim> energy_atoms;

      // Prepare the total number of atoms per cell in this container.
      // The container should also contain the information of total number of
      // atoms per cell for ghost cells on the current MPI process.
      std::map<types::CellIteratorType<dim>, unsigned int> n_atoms_per_cell;

      // Prepare the number of cluster atoms per cell in this container.
      std::map<types::CellIteratorType<dim>, unsigned int> n_cluster_atoms_per_cell;

      // Get the squared_energy_radius to identify energy atoms.
      const double squared_energy_radius =
        dealii::Utilities::fixed_power<2> (WeightsByBase<dim>::maximum_cutoff_radius +
                                           WeightsByBase<dim>::cluster_radius);

      // Get the squared_cluster_radius to identify cluster atoms.
      const double squared_cluster_radius =
        dealii::Utilities::fixed_power<2>(WeightsByBase<dim>::cluster_radius);

      // Loop over all active cells of the mesh and initialize
      // n_atoms_per_cell and n_cluster_atoms_per_cell.
      for (types::CellIteratorType<dim>
           cell  = mesh.begin_active();
           cell != mesh.end();
           cell++)
        {
          n_atoms_per_cell[cell]         = 0;
          n_cluster_atoms_per_cell[cell] = 0;
        }

      // Loop over all atoms, see if a given atom is energy atom and
      // if so if it's a cluster atom.
      // While there, count the total number of atoms per cell and
      // number of cluster atoms per cell.
      for (const auto &cell_atom : atoms)
        {
          const auto &cell         = cell_atom.first;
          Molecule<dim,1> molecule = cell_atom.second;

          Assert (n_atoms_per_cell.find(cell) !=n_atoms_per_cell.end(),
                  ExcMessage("Provided 'mesh' isn't consistent with "
                             "the cell based atoms data structure."));

          n_atoms_per_cell[cell]++;

          // Check the proximity of the atom to it's associated
          // cell's vertices.
          const auto closest_vertex =
            Utilities::find_closest_vertex (molecule.atoms[0].position,
                                            cell);
          if (closest_vertex.second < squared_energy_radius)
            {
              if (closest_vertex.second < squared_cluster_radius)
                {
                  // Increment cluster atom count for this "cell"
                  n_cluster_atoms_per_cell[cell]++;
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
      //---Now update cluster weights with correct value

      // Loop over all the energy atoms,
      // update their weights by multiplying with the factor
      // (n_atoms/n_cluster_atoms)
      for (auto &energy_atom : energy_atoms)
        {

          Assert (n_cluster_atoms_per_cell.at(energy_atom.first) != 0,
                  ExcInternalError());

          // The cluster weight was previously set to 1. if the atom is
          // cluster atom and 0. if the atom is not cluster atom.
          energy_atom.second.cluster_weight *=
            static_cast<double>(n_atoms_per_cell.at(energy_atom.first))
            /
            static_cast<double>(n_cluster_atoms_per_cell.at(energy_atom.first));
        }

      //---Check in Debug mode that n_atoms and n_energy_atoms computed here
      //   are indeed similar to what CellAtomTools functions return.
      //   The reason for not using CellAtomTools functions is that
      //   the code here is already optimized and tested. So we assert that
      //   CellAtomTools functions also yield same result.
#ifdef Debug
      for (types::CellIteratorType<dim>
           cell  = mesh.begin_active();
           cell != mesh.end();
           cell++)
        {
          // Get n_atoms_in_cell using atoms_range.second
          const auto atoms_range =
            CellAtomTools::atoms_range_in_cell(cell, atoms);

          // Get the number of cluster atoms in cell.
          // It is legal to call this function as we have
          // already updated cluster weights
          const auto n_cluster_atoms_in_cell =
            CellAtomTools::n_cluster_atoms_in_cell(cell, energy_atoms);

          Assert (n_atoms_per_cell[cell] == atoms_range.second,
                  ExcInternalError());

          Assert (n_cluster_atoms_per_cell[cell] == n_cluster_atoms_in_cell,
                  ExcInternalError());
        }
#endif

      return energy_atoms;
    }



    // Instantiations.
    template class WeightsByCell<1>;
    template class WeightsByCell<2>;
    template class WeightsByCell<3>;


  } // namespace Cluster


} // namespace dealiiqc

