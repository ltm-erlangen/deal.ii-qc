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
       * Constructor takes in @see ConfigureQC object to initialize atom
       * attributes namely: @see atoms, @see masses and @see atomtype_to_atoms.
       */
      AtomHandler( const ConfigureQC& );

      /**
       * setup atom attributes namely:
       * @see atoms, @see masses and @see atomtype_to_atoms
       * using an istream object.
       */
      void setup( std::istream &);

      /**
       * Initialize or update neighbor lists of the @see energy_atoms.
       * This function can be called as often as one deems necessary.
       */
      void update_neigh_lists();

      // TODO: Temporarily kept here (move to utilities)
      /**
       * Run through all atoms and find cells to which they belong.
       * Due to the assumption of a linear deformation gradient with in a cell
       * in the QC formulation, the atoms assigned to a specific cell continue
       * to live in that cell. Therefore this function will be called only once
       * for a QC simulation.
       */
      void associate_atoms_with_cells (const dealii::Mapping<dim> &,
                                       const dealii::DoFHandler<dim> &);

    protected:

      /**
       * A vector of charges of different atom species.
       */
      std::vector<types::charge> charges;

      /**
       * A vector to store masses of different atom species.
       */
      std::vector<double> masses;

      //TODO: temporarily kept here. move from qc to here?
      /**
       * A vector of atoms in the system. The vector is used only for
       * initializing cell based data structures that would actually be
       * used for computations.
       */
      std::vector<Atom<dim>> atoms;

      /**
       * Neighbourlist
       */
      std::multimap<types::global_atom_index,
                    std::pair<typename DoFHandler<dim>::active_cell_iterator,
                              std::list<types::global_atom_index>>> neigh_list;

  };

} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
