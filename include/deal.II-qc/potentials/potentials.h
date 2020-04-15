
#ifndef __dealii_qc_potentials_h
#define __dealii_qc_potentials_h

#include <deal.II-qc/utilities.h>

#include <utility>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A namespace to define all interaction potentials of the atomistic
 * system and related data members.
 */
namespace Potential
{
  /**
   * An enumeration of all the pair potentials implemented in
   * Potentials namespace.
   */
  enum InteractionTypes
  {
    /**
     * Born-Mayer-Huggins or Tosi/Fumi pair potential.
     * (see PairBornCutManager)
     */
    Born = 0,

    /**
     * Born-Mayer-Huggins with class2 and Coulomb pair potnetial
     * computed using the Wolf summation method
     * (see PairBornCutClass2CoulWolfManager)
     */
    Born_Class2_Coul_Wolf = 1,

    /**
     * The COMPASS class2 pair potential.
     * (see PairClass2Manager)
     */
    Class2 = 2,

    /**
     * Coulomb pair potential computed using Wolf summation method.
     * (see PairCoulWolfManager class).
     */
    Coul_Wolf = 3,

    /**
     * Truncated Lenard-Jones contribution to the potential
     * (see, for example, PairLJCutManager class).
     */
    LJ = 4,

    /**
     * Truncated Lennard-Jones along with Coulomb pair potential computed
     * using the Wolf summation method.
     * (see PairLJCutCoulWolfManager)
     */
    LJ_Coul_Wolf = 5
  };


  /**
   * Return a pair of atom type such that the first element is less than
   * or equal to that of the second element given the two atom types
   * @p i_atom_type and @p j_atom_type.
   */
  inline std::pair<types::atom_type, types::atom_type>
  get_pair(const types::atom_type i_atom_type,
           const types::atom_type j_atom_type)
  {
    return (i_atom_type <= j_atom_type) ?
             std::make_pair(i_atom_type, j_atom_type) :
             std::make_pair(j_atom_type, i_atom_type);
  }

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_potentials_h */
