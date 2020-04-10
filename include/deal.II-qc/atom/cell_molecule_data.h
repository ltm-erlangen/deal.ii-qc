#ifndef __dealii_qc_cell_molecule_data_h
#define __dealii_qc_cell_molecule_data_h

#include <deal.II-qc/atom/molecule.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace types
{
  /**
   * A typedef for a pair of cell and molecule.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMolecule = typename std::pair<CellIteratorType<dim, spacedim>,
                                          Molecule<spacedim, atomicity>>;

  /**
   * A typedef for a const pair of cell and molecule.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using ConstCellMolecule =
    const typename std::pair<CellIteratorType<dim, spacedim>,
                             Molecule<spacedim, atomicity>>;

  /**
   * A typedef for container that holds cell and associated molecules.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMoleculeContainerType =
    typename std::multimap<CellIteratorType<dim, spacedim>,
                           Molecule<spacedim, atomicity>>;

  /**
   * A typedef for iterator over CellMoleculeContainerType.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMoleculeIteratorType =
    typename CellMoleculeContainerType<dim, atomicity, spacedim>::iterator;

  /**
   * A typedef for const_iterator over CellMoleculeContainerType.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMoleculeConstIteratorType =
    typename CellMoleculeContainerType<dim, atomicity, spacedim>::
      const_iterator;

  /**
   * A typedef for a pair of const_iterators over CellMoleculeContainerType
   * which could be used in the case of storing a iterator range.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMoleculeConstIteratorRangeType =
    typename std::pair<CellMoleculeConstIteratorType<dim, atomicity, spacedim>,
                       CellMoleculeConstIteratorType<dim, atomicity, spacedim>>;

  /**
   * A typedef for neighbor lists, a multimap of a pair of cells and a pair of
   * molecules in a cell based data structure.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  using CellMoleculeNeighborLists = typename std::multimap<
    std::pair<ConstCellIteratorType<dim, spacedim>,
              ConstCellIteratorType<dim, spacedim>>,
    std::pair<CellMoleculeConstIteratorType<dim, atomicity, spacedim>,
              CellMoleculeConstIteratorType<dim, atomicity, spacedim>>>;


} // namespace types



/**
 * A principal class that holds cell based molecule data structures with the
 * association between molecules and mesh.
 *
 * The association between molecules and cells allows for accessing each
 * molecule through its associated cell. The cell based molecule data
 * structures in this class can be initialized using
 * CellMoleculeTools::build_cell_molecule_data() and (derived class)
 * implementation of Cluster::WeightsByBase::update_cluster_weights().
 */
template <int dim, int atomicity = 1, int spacedim = dim>
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
   * A multidimensional array to store bonds between different atom stamps.
   */
  types::bond_type bonds[atomicity][atomicity];

  /**
   * The cell based data member that contains cells and molecules
   * association in the Lagrangian (undeformed) configuration of the system.
   * More specifically, #cell_molecules contains association between locally
   * relevant cells and locally relevant molecules in the Lagrangian
   * (undeformed) configuration of the system.
   */
  types::CellMoleculeContainerType<dim, atomicity, spacedim> cell_molecules;

  /**
   * The cell based data member that contains cells and energy molecules
   * association of the system.
   *
   * This data member contains energy molecules with the current atoms'
   * positions. This information is the central information for
   * energy and force computations. Each MPI process would then be
   * responsible to compute energy and forces only of its locally owned
   * sampling (or cluster) molecules.
   */
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
    cell_energy_molecules;
};


DEAL_II_QC_NAMESPACE_CLOSE


#endif // __dealii_qc_cell_molecule_data_h
