#ifndef __dealii_qc_cell_molecule_data_h
#define __dealii_qc_cell_molecule_data_h

#include <deal.II-qc/atom/molecule.h>

namespace dealiiqc
{

  namespace types
  {

    /**
     * A typedef for a pair of cell and molecule.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMolecule =
      typename
      std::pair<CellIteratorType<dim, spacedim>, Molecule<spacedim, atomicity>>;

    /**
     * A typedef for a const pair of cell and molecule.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using ConstCellMolecule =
      const typename
      std::pair<CellIteratorType<dim, spacedim>, Molecule<spacedim, atomicity>>;

    /**
     * A typedef for container that holds cell and associated molecules.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMoleculeContainerType =
      typename std::multimap<CellIteratorType<dim, spacedim>, Molecule<spacedim, atomicity>>;

    /**
     * A typedef for iterator over CellMoleculeContainerType.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMoleculeIteratorType =
      typename CellMoleculeContainerType<dim, atomicity, spacedim>::iterator;

    /**
     * A typedef for const_iterator over CellMoleculeContainerType.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMoleculeConstIteratorType =
      typename CellMoleculeContainerType<dim, atomicity, spacedim>::const_iterator;

    /**
     * A typedef for a pair of const_iterators over CellMoleculeContainerType
     * which could be used in the case of storing a iterator range.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMoleculeConstIteratorRangeType =
      typename
      std::pair
      <
      CellMoleculeConstIteratorType<dim, atomicity, spacedim>,
      CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      >;

    /**
     * A typedef for neighbor lists, a multimap of a pair of cells and a pair of
     * molecules in a cell based data structure.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    using CellMoleculeNeighborLists =
      typename std::multimap <
      std::pair
      <
      ConstCellIteratorType<dim, spacedim>,
      ConstCellIteratorType<dim, spacedim>
      >,
      std::pair
      <
      CellMoleculeConstIteratorType<dim, atomicity, spacedim>,
      CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      >>;


  } // types



  /**
   * A principal class that holds cell based molecule data structures with the
   * association between molecules and mesh.
   *
   * The cell based molecule data structures in this class can be initialized
   * using CellMoleculeDataHandler class member functions.
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  struct CellMoleculeData
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
     * The cell based data structure that contains cells and molecules
     * association for all the molecules in the system.
     *
     * The following optimization technique is employed while associating
     * molecules to cells.
     * For each molecule in the system, before going over all locally owned
     * cells to find if the molecule's location lies within it, we can first
     * check whether the molecule's location lies inside the bounding box of
     * the current processor's set of locally relevant cells.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim> cell_molecules;

    /**
     * The cell based data structure that contains cells and energy molecules
     * association for all the energy molecules in the system needed by a
     * current MPI core. This data member contains the central information for
     * energy and force computations.
     *
     * The function QC::setup_energy_molecules_with_cluster_weights() is
     * responsible for updating this data member.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim> cell_energy_molecules;

  };


} // namespace dealiiqc


#endif // __dealii_qc_cell_molecule_data_h
