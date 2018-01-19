
#ifndef __dealii_qc_atom_cell_molecule_tools_h
#define __dealii_qc_atom_cell_molecule_tools_h


#include<deal.II-qc/atom/cell_molecule_data.h>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A namespace for cell based data structures' utility functions.
 */
namespace CellMoleculeTools
{

  /**
   * Return a pair of range of constant iterators to CellMoleculeContainerType
   * object of atoms @p cell_molecules and the number of atoms of @p atoms in
   * a given @p cell.
   *
   * If no molecules are in the queried cell then the function return the
   * pair of (pair of) end iterators and zero. If @p cell_molecules is an
   * empty CellMoleculeContainer, then the function throws an error that
   * the provided CellMoleculeContainer is empty.
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  std::pair
  <
  types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
  ,
  unsigned int
  >
  molecules_range_in_cell
  (const types::CellIteratorType<dim, spacedim>                     &cell,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules);

  /**
   * Return the number of molecules in @p cell_energy_molecules, associated to
   * @p cell, which have non-zero cluster weights.
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  unsigned int
  n_cluster_molecules_in_cell
  (const types::CellIteratorType<dim, spacedim>                     &cell,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_energy_molecules);

  /**
   * Prepare and return a CellMoleculeData object based on the given @p mesh
   * by parsing the atom data information in @p is by identifying locally
   * relevant cells for each MPI process. All the data members of the returned
   * object are initialized except CellMoleculeData::cell_energy_molecules,
   * which can be updated using any of the derived classes of
   * Cluster::WeightsByBase.
   *
   * <h5>Association between locally relevant cells and molecules</h5>
   *
   * For each MPI process, locally relevant cells (see MoleculeHandler) are
   * identified each MPI process. It is assumed that the underlying
   * triangulation i.e., @p mesh has its ghost cells already setup (see
   * parallel::shared::Triangulation::setup_ghost_cells()). The locally relevant
   * cells are all the cells excpet artificial cells, if any. In other words,
   * it is the union of all the locally owned cells and the ghost cells.
   *
   * To associate locally relevant cells and molecules, each MPI process does
   * the following. For each molecule in the system look for a locally
   * relevant cell of the @p mesh which contains the molecule in the
   * Lagrangian (undeformed) configuration (also referred to as the molecule's
   * initial location which can be obtained from the
   * molecule_initial_location() helper function). If such a cell is found
   * which is locally relevant from the perspective of the current MPI
   * process, the molecule is locally relevant and is associated to the
   * cell and kept by the current MPI process. Otherwise the current MPI
   * process disregards the molecule as it should be picked up by another
   * MPI process.
   *
   * The following optimization technique is employed while associating
   * molecules to cells.
   * For each molecule in the system, before going over all locally relevant
   * cells to find if the molecule's location lies within it, we can first
   * check whether the molecule's location lies inside the bounding box of
   * the current processor's set of locally relevant cells.
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  CellMoleculeData<dim, atomicity, spacedim>
  build_cell_molecule_data (std::istream                       &is,
                            const Triangulation<dim, spacedim> &mesh);

  /**
   * Return molecule number density, the ratio of the total number of molecules
   * associated to the mesh and the volume of the mesh.
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  double
  compute_molecule_density
  (const Triangulation<dim, spacedim>                               &triangulation,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules);

} // namespace CellMoleculeTools


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_atom_cell_molecule_tools_h */
