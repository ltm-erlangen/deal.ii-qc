

#include <dealiiqc/atom/cluster_weights.h>

namespace dealiiqc
{

  namespace Cluster
  {

    //------------------------------------------------------------------------//
    // WeightsByBase

    template <int dim>
    WeightsByBase<dim>::WeightsByBase (const double &cluster_radius)
      :
      cluster_radius(cluster_radius)
    {}



    template <int dim>
    WeightsByBase<dim>::~WeightsByBase()
    {}



    // Instantiations
    template class WeightsByBase<1>;
    template class WeightsByBase<2>;
    template class WeightsByBase<3>;



    //------------------------------------------------------------------------//
    // WeightsByCell

    template <int dim>
    WeightsByCell<dim>::WeightsByCell (const double &cluster_radius)
      :
      WeightsByBase<dim>(cluster_radius)
    {}



    template <int dim>
    void
    WeightsByCell<dim>::update_cluster_weights (const std::map< types::CellIteratorType<dim>, unsigned int> &n_thrown_atoms_per_cell,
                                                types::CellAtomContainerType<dim> &energy_atoms) const
    {
      // Number of cluster atoms per cell
      std::map<typename types::CellIteratorType<dim>, unsigned int> n_cluster_atoms_per_cell;

      // Initialize n_cluster_atoms_per_cell
      for ( typename types::CellAtomContainerType<dim>::iterator unique_key = energy_atoms.begin(); unique_key != energy_atoms.end(); unique_key = energy_atoms.upper_bound(unique_key->first))
        n_cluster_atoms_per_cell[ unique_key->first] = (unsigned int) 0;

      // Loop over all energy_atoms to compute the number of cluster_atoms
      for ( const auto &cell_atom : energy_atoms)
        {
          const auto &cell = cell_atom.first;
          const Atom<dim> &atom  = cell_atom.second;

          // TODO use is_cluster_atom from atom struct
          // TODO When is_cluster_atom, one could remore cluster_radius member variable.
          if ( Utilities::is_point_within_distance_from_cell_vertices( atom.position, cell, WeightsByBase<dim>::cluster_radius) )
            // Increment cluster atom count for this "cell"
            n_cluster_atoms_per_cell[cell]++;
        }

      for ( const auto &cell_count : n_cluster_atoms_per_cell)
        {
          const auto &cell = cell_count.first;
          const double n_cluster_atoms = cell_count.second;

          Assert ( n_thrown_atoms_per_cell.count(cell) > 0,
                   ExcInternalError());

          // The total number of atoms in a cell is the sum of thrown atoms
          // and the energy_atoms in the cell.
          const double n_cell_atoms = n_thrown_atoms_per_cell.at(cell) + energy_atoms.count(cell);

          // Loop over all the energy atoms in the cell,
          // if they are cluster atoms,
          // update their weights (n_cell_atoms/n_cluster_atoms)
          // if they are not cluster atoms,
          // set their weights to zero.
          auto cell_range = energy_atoms.equal_range(cell);
          for ( auto &cell_atom = cell_range.first; cell_atom !=cell_range.second; ++cell_atom)
            {
              Atom<dim> &atom = cell_atom->second;

              if ( Utilities::is_point_within_distance_from_cell_vertices( atom.position, cell, WeightsByBase<dim>::cluster_radius) )
                atom.cluster_weight = n_cell_atoms / n_cluster_atoms;
              else
                atom.cluster_weight = 0.;
            }
        }
    }



    // Instantiations.
    template class WeightsByCell<1>;
    template class WeightsByCell<2>;
    template class WeightsByCell<3>;


  } // namespace Cluster


} // namespace dealiiqc

