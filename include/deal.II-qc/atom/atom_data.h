

#ifndef __dealii_qc_atom_data_h
#define __dealii_qc_atom_data_h

#include <deal.II-qc/atom/atom.h>

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

    /**
     * A typedef for a pair of const_iterators over CellAtomContainerType which
     * could be used in the case of storing a iterator range.
     */
    template<int dim>
    using CellAtomConstIteratorRangeType = typename std::pair<CellAtomConstIteratorType<dim>, CellAtomConstIteratorType<dim>>;

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
     * The cell based data structure that contains cells and atoms association
     * for all the atoms in the system. The function
     * AtomHandler::parse_atoms_and_assign_to_cells() is responsible for
     * updating this data member.
     *
     * The following optimization technique is employed while adding atoms.
     * For each atom in the system, before going over all locally owned
     * cells to find if the atom lies within it, we can first check whether the
     * atom's location lies inside the bounding box of the current processor's
     * set of locally relevant cells.
     */
    types::CellAtomContainerType<dim> atoms;

    /**
     * The cell based data structure that contains cells and energy atoms
     * association for all the energy atoms in the system needed by a
     * current MPI core. This data member contains the central information for
     * energy and force computations.
     *
     * The function QC::setup_energy_atoms_with_cluster_weights() is
     * responsible for updating this data member.
     */
    types::CellAtomContainerType<dim> energy_atoms;

  };


} // namespace dealiiqc


#endif /* __dealii_qc_atom_data_h */
