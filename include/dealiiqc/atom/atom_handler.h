#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h

#include <deal.II/grid/grid_tools.h>

#include <dealiiqc/atom/atom_data.h>
#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/io/parse_atom_data.h>
#include <dealiiqc/utilities.h>

namespace dealiiqc
{

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
    void parse_atoms_and_assign_to_cells( const types::MeshType<dim> &mesh,
                                          AtomData<dim> &atom_data);

    /**
     * Return the neighbor lists of the @see energy_atoms.
     * This function can be called as often as one deems necessary.
     */
    std::multimap< std::pair< types::ConstCellIteratorType<dim>, types::ConstCellIteratorType<dim>>, std::pair< types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> > >
    get_neighbor_lists( const types::CellAtomContainerType<dim> &energy_atoms);


  protected:

    /**
     * A constant reference to ConfigureQC object
     */
    const ConfigureQC &configure_qc;

    // TODO: Remove below members.
    //       Removing will make significant changes to update_neighbor_lists()
    //       I will make these changes in the next PR.
    /**
     * Neighbor lists using cell approach.
     * For each cell loop over all nearby relevant cells only once
     * and loop over all interacting atoms between the two cells.
     */
    std::multimap< std::pair< types::ConstCellIteratorType<dim>, types::ConstCellIteratorType<dim>>, std::pair< types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> > > neighbor_lists;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
