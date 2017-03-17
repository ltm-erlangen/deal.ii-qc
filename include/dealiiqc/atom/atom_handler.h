#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h

#include <deal.II/grid/grid_tools.h>

#include <dealiiqc/atom/atom.h>
#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/io/parse_atom_data.h>
#include <dealiiqc/utility.h>

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
     * Constructor takes in @p configure_qc object to initialize atom
     * attributes namely: @see atoms, @see masses and @see atomtype_to_atoms.
     */
    AtomHandler( const ConfigureQC &configure_qc);

    // TODO: Rework in the future to not have a vector of atoms data
    /**
     * setup atom attributes namely:
     * @see atoms, @see masses and @see atomtype_to_atoms
     * Run through all atoms and find cells to which they belong.
     */
    void parse_atoms_and_assign_to_cells( const parallel::shared::Triangulation<dim> &tria);

    /**
     * Initialize or update neighbor lists of the @see energy_atoms.
     * This function can be called as often as one deems necessary.
     */
    void update_neighbor_lists();

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    typedef typename parallel::shared::Triangulation<dim>::active_cell_iterator CellIterator;

    /**
     * A typedef for iterating over atoms
     */
    typename std::multimap< CellIterator, Atom<dim>>::const_iterator CellAtomIterator;

  protected:

    /**
     * ConfigureQC object for initializing @see atoms, @see charges, @see masses
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
    std::multimap< CellIterator, Atom<dim>> atoms;

    /**
     * Neighbor lists using cell approach.
     * For each cell loop over all nearby relevant cells only once
     * and loop over all interacting atoms between the two cells.
     */
    std::multimap<CellIterator,
        std::multimap<CellIterator, std::pair<CellAtomIterator, CellAtomIterator>>> neighbor_lists;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
