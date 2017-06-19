#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h

#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/atom/atom_data.h>
#include <deal.II-qc/atom/parse_atom_data.h>
#include <deal.II-qc/configure/configure_qc.h>

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
     * Setup few data members in AtomData (namely: AtomData::atoms,
     * AtomData::masses and AtomData::atomtype_to_atoms) by parsing the atom
     * data information provided through #configure_qc. This function does not
     * update AtomData::energy_atoms.
     *
     * For each and every atom in the system, find the locally relevant cell
     * of the @p mesh which surrounds it. If the atom doesn't belong to
     * any of the locally relevant cells, it is thrown. In the case when
     * a locally relevant cell is found is is inserted into AtomData::atoms.
     *
     * A locally relevant cell is one which is either a locally owned cell or
     * a ghost cell within a ConfigureQC::ghost_cell_layer_thickness from the
     * locally owned cell. A ghost cell could contain atoms whose
     * positions are needed for computing energy or forces on locally owned
     * cluster atoms. An MPI process computes energy and forces only of it's
     * locally relevant cluster atoms.
     *
     * All the atoms which are outside the locally relevant cells are thrown
     * away.
     */
    void parse_atoms_and_assign_to_cells( const types::MeshType<dim> &mesh,
                                          AtomData<dim> &cell_molecule_data) const;

    /**
     * Return the neighbor lists of AtomData::energy_atoms.
     *
     * This function can be called as often as one deems necessary. The function
     * currently assumes that the deformation is small that the neighbor lists
     * are exactly the same as that of reference (undeformed) configuration.
     */
    std::multimap< std::pair<types::ConstCellIteratorType<dim>, types::ConstCellIteratorType<dim>>, std::pair<types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> > >
        get_neighbor_lists( const types::CellAtomContainerType<dim> &cell_energy_molecules) const;


  protected:

    /**
     * A constant reference to ConfigureQC object
     */
    const ConfigureQC &configure_qc;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
