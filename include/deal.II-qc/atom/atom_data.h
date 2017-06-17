

#ifndef __dealii_qc_atom_data_h
#define __dealii_qc_atom_data_h

#include <deal.II-qc/atom/cell_molecule_data.h>

namespace dealiiqc
{

  namespace types
  {
    /**
     * A typedef for mesh.
     */
    template<int dim>
    using MeshType = dealii::DoFHandler<dim>;

    // TODO: Remove all of the following type definitions.
    /**
     * A typedef for container that holds cell and associated atoms
     */
    template<int dim>
    using CellAtomContainerType =
      typename types::CellMoleculeContainerType<dim, 1>;

    /**
     * A typedef for iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType =
      typename types::CellMoleculeContainerType<dim, 1>::iterator;

    /**
     * A typedef for const_iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomConstIteratorType =
      typename types::CellMoleculeContainerType<dim, 1>::const_iterator;

    /**
     * A typedef for a pair of const_iterators over CellAtomContainerType which
     * could be used in the case of storing a iterator range.
     */
    template<int dim>
    using CellAtomConstIteratorRangeType =
      typename std::pair<CellAtomConstIteratorType<dim>, CellAtomConstIteratorType<dim>>;

  } // types


  // TODO: Remove the following type definition.
  /**
   * Primary class that holds cell based atom data structures with the
   * association between atoms and mesh.
   */
  template<int dim>
  using AtomData = CellMoleculeData<dim,1>;


} // namespace dealiiqc


#endif /* __dealii_qc_atom_data_h */
