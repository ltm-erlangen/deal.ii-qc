#ifndef __dealii_qc_molecule_h
#define __dealii_qc_molecule_h

#include <deal.II-qc/atom/atom.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * A class for molecules embedded in a <tt>spacedim</tt>-dimensional space.
 *
 * The <tt>atomicity</tt> of a molecule, that is the number of atoms it
 * contains, must be known at compile time.
 */
template<int spacedim, int atomicity=1>
struct Molecule
{

  /**
   * Molecule's global index in the atomistic system.
   */
  types::global_molecule_index global_index;

  /**
   * A list of atoms that constitute this molecule. The size of the list is
   * equal to the atomicity of the molecule. Each atom in the molecule
   * is given a different stamp and therefore the order of atoms is important
   * and is according to their stamp.
   */
  std::array<Atom<spacedim>, atomicity> atoms;

  /**
   * Local index of the molecule within the cell it is associated to.
   *
   * @note During constructing FEValues object for the cell associated to this
   * molecule, the local_index of this molecule can be used as the index for
   * the quadrature point corresponding to this molecule. It is convenient to
   * have the index of the quadrature point while calling the value function
   * of the FEValues object.
   */
  unsigned int local_index;

  /**
   * Position of the molecule in reference coordinates of the cell to which
   * it is associated to.
   */
  Point<spacedim> position_inside_reference_cell;

  /**
   * Representativeness of this molecule in its contribution to the
   * total energy of the atomistic system.
   *
   * A molecule is inside a cluster if any of its atoms are geometrically
   * inside the cluster. Any molecule that is located inside a cluster is a
   * cluster molecule or sampling molecule. All the sampling molecules have
   * non-zero @p cluster_weight.
   */
  double cluster_weight;

};



// TODO: Move the following functions to a different place?
/**
 * Return the initial location of @p molecule. The initial location is
 * nothing but its location in the Lagrangian (undeformed) configuration of
 * the atomistic system. By convention, the initial location of the molecule
 * is the initial position of the first atom of the molecule.
 *
 * @note It is assumed here that the atoms of the molecule are already sorted
 * according to their stamps.
 */
template<int spacedim, int atomicity>
inline
Point<spacedim> molecule_initial_location (const Molecule<spacedim, atomicity> &molecule)
{
  Assert (molecule.atoms.size()>0, ExcInternalError());

#ifdef DEBUG
  for (int a=0; a<atomicity; a++)
    for (int b=a+1; b<atomicity; b++)
      Assert(molecule.atoms[a].type <= molecule.atoms[b].type,
             ExcMessage("Atoms in the molecule are not sorted according "
                        "to their atom types."));
#endif

  return molecule.atoms[0].initial_position;
}



/**
 * Return the least distance squared between atoms of @p molecule_A and
 * @p molecule_B.
 */
template<int spacedim, int atomicity>
inline
double
least_distance_squared (const Molecule<spacedim, atomicity> &molecule_A,
                        const Molecule<spacedim, atomicity> &molecule_B)
{
  double squared_distance = std::numeric_limits<double>::max();

  for (const auto &atom_A : molecule_A.atoms)
    for (const auto &atom_B : molecule_B.atoms)
      {
        const double tmp_squared_distance =
          atom_A.position.distance_square(atom_B.position);

        if (tmp_squared_distance < squared_distance)
          squared_distance = tmp_squared_distance;
      }

  return squared_distance;
}


DEAL_II_QC_NAMESPACE_CLOSE


#endif // __dealii_qc_molecule_h
