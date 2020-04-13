
#ifndef __dealii_qc_pair_base_h
#define __dealii_qc_pair_base_h

#include <deal.II-qc/potentials/potentials.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * Base class of the pair potential classes.
   */
  class PairBaseManager
  {
  public:
    /**
     * Default constructor.
     */
    PairBaseManager();

    /**
     * Virtual destructor.
     */
    virtual ~PairBaseManager();

    /**
     * Set the charges of each atom type in the system using a shared pointer
     * to a list of @p charges. This is needed by the
     * derived classes to compute Coulomb interaction energy when there are
     * atoms with non-zero charges in the system.
     *
     * In typical usage case we do not know all the charges until we parse
     * the atom data. However, we want to create an object of the derived
     * class before we start parsing the atom data. Therefore the
     * charges in system are set-up in a separate function as opposed to the
     * constructor of derived classes.
     *
     * @note This virtual function shares ownership
     * (with the member variable PairBaseManager::charges) over
     * the resources of @p charges_, increasing the use count by one.
     */
    virtual void
    set_charges(std::shared_ptr<std::vector<types::charge>> &charges_);

    /**
     * Return whether the potential is a bond potential or
     * has an augmented bond potential.
     */
    virtual bool
    is_or_has_bond_style() const;

    /**
     * Declare the type of interaction between the atom types @p i_atom_type
     * and @p j_atom_type to be @p interaction through @p parameters.
     */
    virtual void
    declare_interactions(const types::atom_type     i_atom_type,
                         const types::atom_type     j_atom_type,
                         InteractionTypes           interaction,
                         const std::vector<double> &parameters) = 0;

  protected:
    /**
     * A shared pointer to the list of charges \f$q_i\f$ of the different
     * atom typesin the system.
     */
    std::shared_ptr<const std::vector<types::charge>> charges;
  };

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_pair_lj_cut_h */
