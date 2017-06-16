#ifndef __dealii_qc_cell_molecule_data_h
#define __dealii_qc_cell_molecule_data_h

#include <deal.II-qc/atom/molecule.h>

namespace dealiiqc
{

  namespace types
  {

    /**
     * A typedef for container that holds cell and associated molecules.
     */
    template<int dim, int spacedim=dim>
    using CellMoleculeContainerType =
      typename std::multimap<CellIteratorType<dim>, Molecule<spacedim>>;

    /**
     * A typedef for iterator over CellMoleculeContainerType.
     */
    template<int dim, int spacedim=dim>
    using CellMoleculeIteratorType =
      typename std::multimap<CellIteratorType<dim>, Molecule<spacedim>>::iterator;

    /**
     * A typedef for const_iterator over CellMoleculeContainerType.
     */
    template<int dim, int spacedim=dim>
    using CellMoleculeConstIteratorType =
      typename std::multimap<CellIteratorType<dim>, Molecule<spacedim>>::const_iterator;

    /**
     * A typedef for a pair of const_iterators over CellMoleculeContainerType
     * which could be used in the case of storing a iterator range.
     */
    template<int dim, int spacedim=dim>
    using CellMoleculeConstIteratorRangeType =
      typename std::pair<CellMoleculeConstIteratorType<dim,spacedim>, CellMoleculeConstIteratorType<dim,spacedim>>;

  } // types


  /**
   * Primary class that holds cell based molecule data structures with the
   * association between molecules and mesh.
   */
  template<int stamps, int int dim, int spacedim=dim>
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

    // TODO: Change which function will update this member
    /**
     * The cell based data structure that contains cells and molecules
     * association for all the molecules in the system. The function
     * AtomHandler::parse_atoms_and_assign_to_cells() is responsible for
     * updating this data member.
     *
     * The following optimization technique is employed while associating
     * molecules to cells.
     * For each molecule in the system, before going over all locally owned
     * cells to find if the molecule's location lies within it, we can first
     * check whether the molecule's location lies inside the bounding box of
     * the current processor's set of locally relevant cells.
     */
    types::CellMoleculeContainerType<dim, spacedim> cell_molecules;

    /**
     * The cell based data structure that contains cells and energy molecules
     * association for all the energy molecules in the system needed by a
     * current MPI core. This data member contains the central information for
     * energy and force computations.
     *
     * The function QC::setup_energy_molecules_with_cluster_weights() is
     * responsible for updating this data member.
     */
    types::CellMoleculeContainerType<dim, spacedim> cell_energy_molecules;

  };


} // namespace dealiiqc


#endif // __dealii_qc_cell_molecule_data_h
