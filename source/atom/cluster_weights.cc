

#include <dealiiqc/atom/cluster_weights.h>

namespace dealiiqc
{

  namespace Cluster
  {
    //------------------------------------------------------------------------//
    // WeightsByBase

    template<int dim>
    WeightsByBase<dim>::WeightsByBase( const ConfigureQC &config)
      :
      config(config)
    {}

    template class WeightsByBase<1>;
    template class WeightsByBase<2>;
    template class WeightsByBase<3>;


    //------------------------------------------------------------------------//
    // WeightsByCell
    template<int dim>
    WeightsByCell<dim>::WeightsByCell(const ConfigureQC &config)
      :
      WeightsByBase<dim>(config)
    {}


    template<int dim>
    void
    WeightsByCell<dim>::update_cluster_weights( const std::map< typename AtomHandler<dim>::CellIteratorType, unsigned int> &n_thrown_atoms_per_cell,
                                                typename AtomHandler<dim>::CellAtomContainerType &energy_atoms)
    {
      const double cluster_radius = WeightsByBase<dim>::config.get_cluster_radius();

      // Number of cluster atoms count per cell
      std::map<typename AtomHandler<dim>::CellIteratorType, unsigned int> n_cluster_atoms_per_cell;

      // Initialize n_cluster_atoms_per_cell
      for ( auto unique_key = energy_atoms.begin(); unique_key != energy_atoms.end(); unique_key = energy_atoms.upper_bound(unique_key->first))
        n_cluster_atoms_per_cell[ unique_key->first] = (unsigned int) 0;

      // Loop over all energy_atoms to compute the number of cluster_atoms
      for ( const auto &cell_atom : energy_atoms)
        {
          const auto &cell = cell_atom.first;
          const Atom<dim> &atom  = cell_atom.second;

          // There are no locally relevant cluster atoms in ghost cells
          if ( cell->is_locally_owned())
            //TODO use is_cluster_atom from atom struct
            if ( Utilities::is_point_within_distance_from_cell_vertices( atom.position, cell, cluster_radius) )
              // Increment cluster atom count for this "cell"
              ++n_cluster_atoms_per_cell.at(cell);
        }

      for ( const auto &cell_count : n_cluster_atoms_per_cell)
        {
          const auto &cell = cell_count.first;
          for ( auto &cell_atom : energy_atoms )
            {
              Atom<dim> &atom = cell_atom.second;

              if ( Utilities::is_point_within_distance_from_cell_vertices( atom.position, cell, cluster_radius) )
                {
                  // The total number of atoms in a cell is the sum of thrown atoms
                  // and the energy_atoms in the cell.
                  // cluster_weight = n_atoms / n_cluster_atoms;
                  if ( n_thrown_atoms_per_cell.count(cell) )
                    // Some of the atom might have been thrown from cell
                    atom.cluster_weight = static_cast<double>(n_thrown_atoms_per_cell.at(cell) + energy_atoms.count(cell)) /
                                          static_cast<double>(cell_count.second);
                  else
                    // None of the atoms in the cell were thrown
                    atom.cluster_weight = static_cast<double>(energy_atoms.count(cell)) /
                                          static_cast<double>(cell_count.second);
                }
            }
        }
    }

    template class WeightsByCell<1>;
    template class WeightsByCell<2>;
    template class WeightsByCell<3>;


  }

}
