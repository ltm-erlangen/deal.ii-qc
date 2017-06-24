#ifndef __dealii_qc_molecule_handler_h
#define __dealii_qc_molecule_handler_h

#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/configure/configure_qc.h>

namespace dealiiqc
{

  /**
   * A class to manage the cell based molecule data of the system
   * (see CellMoleculeData).
   *
   * deal.II-qc can use multiple machines connected via MPI to parallelize
   * computations of (quasicontinuum approximation of) energy and its gradient.
   *
   * <h3>Local ownership</h3>
   *
   * In an MPI context, when a parallel::shared::Triangulation
   * is employed (which keeps the information of the entire mesh on each
   * process) the concept of a local ownership is used.
   * In a parallel::shared::Triangulation, each cell is owned by exactly one
   * MPI process leading to a partitioning of the underlying mesh into disjoint
   * locally owned cells among MPI processes. The cells stored on an MPI process
   * that are not owned by this process are called ghost cells of the MPI
   * process.
   *
   * <h3>Cells and molecules association</h3>
   *
   * In order to define kinematic constraints on lattice site positions,
   * we need to associate cells and molecules. The association between a cell
   * and molecule is done in the following way. In the Lagrangian (undeformed)
   * configuration of the system given the location of a molecule, we find a
   * cell which contains it. After such an association the concept of local
   * ownership also applies to the molecules of the system.
   *
   * <h3>Locally relevant cells and molecules</h3>
   *
   * Each MPI process is responsible for computation of energy (and its
   * gradient) of the locally owned molecules (or a subset of locally owned
   * molecule by applying certain sampling rules). Molecules which are not
   * owned by an MPI process are called ghost molecules. All the molecules
   * which interact with the locally owned molecules of an MPI process are
   * called locally relevant molecules and their associated cells are the
   * locally relevant cells of the MPI process. Therefore, each MPI process
   * needs to store the updated copies of all locally relevant molecules.
   *
   * To limit memory usage and communication overhead, we would want to limit
   * the number of ghost molecules stored in MPI processes. For this reason,
   * we assume that it is enough to find locally relevant, ghost cells and
   * molecules of each process with in a distance of
   * ConfigureQC::ghost_cell_layer_thickness from the locally owned cells only
   * once in Lagrangian (undeformed) configuration of the system. Which
   * certainly holds for the case of small deformations.
   * For all other cases, it can be done using careful or custom partitioning
   * of the triangulation and choosing sufficiently large
   * ConfigureQC::ghost_cell_layer_thickness.
   *
   * It can also be stated that a locally relevant cell is one which is either
   * a locally owned cell or a ghost cell within a
   * ConfigureQC::ghost_cell_layer_thickness (which is greater than the
   * interaction radius between molecules) from the locally owned cells of an
   * MPI process.
   *
   * (refer to deal.II documentation for further details about
   * parallel::shared::Triangulation).
   */
  template<int dim, int atomicity=1, int spacedim=dim>
  class MoleculeHandler
  {
  public:

    /**
     * Constructor takes in a ConfigureQC object @p configure_qc.
     */
    MoleculeHandler (const ConfigureQC &configure_qc);

    /**
     * Return the neighbor lists of @p cell_energy_molecules.
     *
     * This function can be called as often as one deems necessary. The function
     * currently assumes that the deformation is small that the neighbor lists
     * are exactly the same as that of reference (undeformed) configuration.
     */
    types::CellMoleculeNeighborLists<dim, atomicity, spacedim>
    get_neighbor_lists (const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_energy_molecules) const;


  protected:

    /**
     * A constant reference to ConfigureQC object
     */
    const ConfigureQC &configure_qc;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_molecule_handler_h
