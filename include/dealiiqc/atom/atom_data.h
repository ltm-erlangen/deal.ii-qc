

#ifndef __dealii_qc_atom_data_h
#define __dealii_qc_atom_data_h

#include <dealiiqc/atom/atom.h>

namespace dealiiqc
{

  namespace types
  {
    /**
     * A typedef for mesh.
     */
    template<int dim>
    using MeshType = dealii::DoFHandler<dim>;

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    template<int dim>
    using CellIteratorType = typename MeshType<dim>::active_cell_iterator;

    /**
     * A typedef for container that holds cell and associated atoms
     */
    template<int dim>
    using CellAtomContainerType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >;

    /**
     * A typedef for iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >::iterator;

    /**
     * A typedef for const_iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >::iterator;

  } // types


  /**
   * Primary class that holds atom data and the association between atoms
   * and mesh.
   */
  template<int dim>
  struct AtomData
  {

    /**
     * A vector of charges of different atom species.
     */
    std::vector<types::charge> charges;

    /**
     * A vector to store masses of different atom species.
     */
    std::vector<double> masses;

    /**
     * A lookup data structure for all atoms in the system needed by a
     * current MPI core, namely a union of locally owned and ghost atoms.
     * Used for initializing cell based data structures that would actually be
     * used for computations.
     *
     * Optimization technique (not yet implemented):
     * Before going over all locally owned cells to find if a given atom lies within it,
     * we can first check whether the atom's location lies inside the certain bounding box of the
     * current processor's set of locally owned cells. The bounding box needs to be extended
     * with @see cluster_radius + @see cutoff_radius.
     */
    std::multimap< types::CellIteratorType<dim>, Atom<dim>> energy_atoms;

    /**
     * Neighbor lists using cell approach.
     * For each cell loop over all nearby relevant cells only once
     * and loop over all interacting atoms between the two cells.
     */
    std::multimap< std::pair< types::CellIteratorType<dim>, types::CellIteratorType<dim>>, std::pair< types::CellAtomIteratorType<dim>, types::CellAtomIteratorType<dim> > > neighbor_lists;

    /**
     * Number of locally relevant non-energy atoms per cell.
     * This is exactly the number of non-energy atoms for whom a
     * locally relevant cell is found while updating @see energy_atoms.
     * They were thrown because they weren't energy atoms.
     *
     * @note The map also contains the information of number of
     * thrown atoms per cell for ghost cells on the current
     * MPI process.
     */
    std::map< types::CellIteratorType<dim>, unsigned int> n_thrown_atoms_per_cell;

  };


}


#endif /* __dealii_qc_atom_data_h */
