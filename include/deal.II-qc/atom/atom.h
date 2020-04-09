#ifndef __dealii_qc_atom_h
#define __dealii_qc_atom_h

#include <deal.II/base/point.h>

#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * A class for atoms embedded in a <tt>spacedim</tt>-dimensional space.
 *
 * This class does not contain some more atom attributes such as charge and
 * mass etc. This is because the number of different atom types in an
 * atomistic system is far less than the number of atoms. The charges and
 * masses of different atom types can be stored elsewhere in a central pool.
 *
 * A drude particle can be constructed using two Atom objects. One object for
 * the charged core and another for the charged shell.
 */
template <int spacedim>
struct Atom
{
  /**
   * Global atom index of the atom.
   */
  types::global_atom_index global_index;

  /**
   * Atom type of the atom.
   */
  types::atom_type type;

  /**
   * Initial position of the atom in <tt>spacedim</tt>-dimensional space.
   */
  Point<spacedim> initial_position;

  /**
   * Current position of the atom in <tt>spacedim</tt>-dimensional space.
   */
  Point<spacedim> position;
};


DEAL_II_QC_NAMESPACE_CLOSE

#endif // __dealii_qc_atom_h
