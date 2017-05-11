

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
     * A typedef for active_cell_iterator for ease of use
     */
    template<int dim>
    using ConstCellIteratorType = const typename MeshType<dim>::active_cell_iterator;

    /**
     * A typedef for container that holds cell and associated atoms
     */
    template<int dim>
    using CellAtomContainerType = typename std::multimap<CellIteratorType<dim>, Atom<dim> >;

    /**
     * A typedef for iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType = typename std::multimap<CellIteratorType<dim>, Atom<dim> >::iterator;

    /**
     * A typedef for const_iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomConstIteratorType = typename std::multimap<CellIteratorType<dim>, Atom<dim> >::const_iterator;


  } // types


  /**
   * Primary class that holds cell based atom data structures with the
   * association between atoms and mesh.
   */
  template<int dim>
  struct AtomData
  {

    /**
     * A vector of charges of different atom species.
     */
    std::shared_ptr<std::vector<types::charge>> charges;

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
     * The following optimization technique is employed while adding energy
     * atoms. For each atom in the system, before going over all locally owned
     * cells to find if the atom lies within it, we can first check whether the
     * atom's location lies inside the bounding box of the current processor's
     * set of locally relevant cells.
     */
    std::multimap<types::CellIteratorType<dim>, Atom<dim>> energy_atoms;

    /**
     * The number of locally relevant non-energy atoms per cell.
     * This is exactly the number of non-energy atoms for whom a
     * locally relevant cell is found while updating #energy_atoms.
     * They were thrown because they weren't energy atoms.
     *
     * @note The map also contains the information of number of
     * thrown atoms per cell for ghost cells on the current
     * MPI process.
     */
    std::map<types::CellIteratorType<dim>, unsigned int> n_thrown_atoms_per_cell;

  };


}


#endif /* __dealii_qc_atom_data_h */
