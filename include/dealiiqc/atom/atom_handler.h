#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
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

    /**
     * A typedef for mesh.
     */
    using MeshType = dealii::DoFHandler<dim>;

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    using CellIterator = typename MeshType::active_cell_iterator;

    /**
     * A typedef for cell and atom associations
     */
    using CellAtoms = typename std::multimap< CellIterator, Atom<dim>>;

    /**
     * A typedef for iterating over @see CellAtoms
     */
    using CellAtomsIterator = typename std::multimap< CellIterator, Atom<dim>>::iterator;

    /**
     * A typedef for iterating over @see energy_atoms
     */
    using ClusterAtomsIterator = typename std::multimap< CellIterator, CellAtomsIterator>;

    // TODO: Rework in the future to not have a vector of atoms data
    /**
     * setup atom attributes namely:
     * @see atoms, @see masses and @see atomtype_to_atoms
     * Run through all atoms and find cells to which they belong.
     */
    void parse_atoms_and_assign_to_cells( const MeshType &mesh,
                                          const MPI_Comm &comm);

    /**
     * Initialize or update neighbor lists of the @see energy_atoms.
     * This function can be called as often as one deems necessary.
     */
    void update_neighbor_lists();

    /**
     * Write out the information of number of atoms, number of energy atoms
     * and number of cluster atoms per cell.
     * TODO: Extend this to output more cell data?
     * TODO: Make the function MPI compliant!
     */
    void write_cell_data(const MeshType &mesh,
                         std::ostream &out);

  protected:

    /**
     * Return true of an atom is cluster atom.
     * Should be used only after updating atom parent cell attribute.
     */
    inline
    bool is_cluster_atom( const Atom<dim> &a);

    /**
     * Return true of an atom is energy atom.
     * Should be used only after updating atom parent cell attribute.
     */
    inline
    bool is_energy_atom( const Atom<dim> &a);

    /**
     * Update energy atoms
     */
    void update_energy_atoms();

    /**
     * Update cluster weights
     */
    void update_cluster_weights();

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
     * A multimap container to store atoms that contribute towards energy
     * computations for the current MPI core. Each energy atom is stored in the
     * container along with its cell.
     */
    CellAtoms energy_atoms;

    /**
     * A multimap container to store atoms that are inside clusters.
     */
    ClusterAtomsIterator cluster_atoms_iterator;

    /**
     * Neighbor lists for cells.
     */
    std::multimap<CellIterator, CellIterator> cell_neighbor_lists;

    /**
     * Neighbor lists using cell approach.
     * For each cell loop over all nearby relevant cells only once
     * and loop over all interacting atoms between the two cells.
     */
    typename
    std::multimap< std::pair< CellIterator, CellIterator>,
        std::pair< CellAtomsIterator, CellAtomsIterator> > atom_neighbor_lists;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
