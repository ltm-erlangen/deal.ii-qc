

#include <dealiiqc/potentials/pair_lj_cut.h>

namespace dealiiqc
{

  namespace Potential
  {
    template<int dim>
    PairLJCut<dim>::PairLJCut( const ConfigureQC &configure)
    :
    Pair<dim>(configure)
    {}

    template<int dim>
    double
    PairLJCut<dim>::value ( const types::NeighborListsType<dim> &neighbor_lists)
    {
      double energy = 0.;

      // NeighborListType is arranged as cell_pair, atom_pair
      for ( const auto &cell_pair_atom_pair : neighbor_lists )
        {
          const Atom<dim> &
          atom_I = (cell_pair_atom_pair.second.first)->second,
          atom_J = (cell_pair_atom_pair.second.second)->second;

          // TODO: implement LJ code here
          //       now has dummy code.
          energy += (atom_I-atom_J).norm();
          // TODO: update energy_per_cluster_atom
        }

      return energy;
    }

    template<int dim>
    void
    PairLJCut<dim>::gradient ( vector_t &gradient,
                               const types::CellAssemblyData<dim> &cell_assembly_data,
                               const types::NeighborListsType<dim> &neighbor_lists) const
    {
      // TODO
    }

    template<int dim>
    void
    PairLJCut<dim>::parse_pair_coefficients ()
    {
      // TODO
    }

  }

} /* namespace dealiiqc */
