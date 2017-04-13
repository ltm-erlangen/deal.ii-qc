#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h

#include <deal.II/grid/grid_tools.h>

#include <dealiiqc/atom/atom.h>
#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/io/parse_atom_data.h>
#include <dealiiqc/utilities.h>

namespace dealiiqc
{

  // TODO: easy access to atoms in any cell using atom iterators per cell
  /**
   * Manage initializing and indexing the atoms, distributing them to cells so
   * as that the cells own disjoint atom index sets.
   */
  template<int dim>
  class AtomHandler
  {
  public:

    /**
     * Constructor takes in @p configure_qc object
     * to initialize atom attributes through the class's
     * member function parse_atoms_and_assign_to_cells().
     */
    AtomHandler( const ConfigureQC &configure_qc);

    /**
     * A typedef for mesh.
     */
    using MeshType = dealii::DoFHandler<dim>;

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    using CellIteratorType = typename MeshType::active_cell_iterator;

    /**
     * A typedef for cell and atom associations
     */
    using CellAtomContainerType = typename std::multimap< CellIteratorType, Atom<dim>>;

    /**
     * A typedef for iterating over @see CellAtoms
     */
    using CellAtomIteratorType = typename std::multimap< CellIteratorType, Atom<dim>>::iterator;

    /**
     * Setup atom attributes namely:
     * @see energy_atoms, @see masses and @see atomtype_to_atoms
     *
     * For each and every atom in the system, find the locally relevant cell
     * of the @p mesh which surrounds it. If the atom doesn't belong to
     * any of the locally relevant cells, it is thrown. In the case when
     * a locally relevant cell is found, and if the atom is energy atom it is
     * inserted into @see energy_atoms.
     *
     * A locally relevant cell is one which is either a locally owned cell or
     * a ghost cell within a certain distance(determined by cluster radius and
     * interaction radius). A ghost cell could contain atoms whose positions are
     * relevant for computing energy or forces on locally relevant
     * cluster atoms. An MPI process computes energy and forces only of it's
     * locally relevant cluster atoms.
     *
     * All the atoms which are not locally relevant energy_atoms
     * are thrown away. However, a count of the number of (locally relevant)
     * non-energy atoms (i.e., for which a locally relevant cell is found
     * but are not energy atoms) is kept using @see n_thrown_atoms_per_cell
     * for the sake of updating cluster weights.
     *
     */
    void parse_atoms_and_assign_to_cells( const MeshType &mesh);

    /**
     * Initialize or update neighbor lists of the @see energy_atoms.
     * This function can be called as often as one deems necessary.
     */
    void update_neighbor_lists();


  protected:

    /**
     * A constant reference to ConfigureQC object
     */
    const ConfigureQC &configure_qc;

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
    std::multimap< CellIteratorType, Atom<dim>> energy_atoms;

    /**
     * Neighbor lists using cell approach.
     * For each cell loop over all nearby relevant cells only once
     * and loop over all interacting atoms between the two cells.
     */
    std::multimap< std::pair<CellIteratorType, CellIteratorType>, std::pair<CellAtomIteratorType, CellAtomIteratorType> > neighbor_lists;

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
    std::map<CellIteratorType, unsigned int> n_thrown_atoms_per_cell;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
