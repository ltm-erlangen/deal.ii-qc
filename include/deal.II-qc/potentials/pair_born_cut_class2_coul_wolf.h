
#ifndef __dealii_qc_pair_born_cut_class2_coul_wolf_h
#define __dealii_qc_pair_born_cut_class2_coul_wolf_h

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_born_cut.h>
#include <deal.II-qc/potentials/pair_class2.h>
#include <deal.II-qc/potentials/pair_coul_wolf.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * PairBornCutManager, PairClass2Manager along with PairCoulWolfManager.
   * Consequently, the energy and gradient values computed using this class is
   * the sum of the three values computed individually.
   */
  class PairBornCutClass2CoulWolfManager : public PairBaseManager
  {
  public:
    /**
     * Constructor that takes in the damping coefficient @p alpha,
     * the cutoff radius @p coul_cutoff_radius to be used for computing Coulomb
     * energy contribution using Wolf summation and the cutoff radius
     * @p born_cutoff_radius for computing born potential contribution.
     * The use of @p factor_coul is explained in
     * PairCoulWolfManager::PairCoulWolfManager().
     */
    PairBornCutClass2CoulWolfManager(const double &alpha,
                                     const double &coul_cutoff_radius,
                                     const double &born_cutoff_radius,
                                     const double &factor_coul = 1.);

    /**
     * @copydoc PairBaseManager::set_charges().
     */
    void
    set_charges(std::shared_ptr<std::vector<types::charge>> &charges_) override;

    /**
     * Declare the type of interaction between the atom types @p i_atom_type
     * and @p j_atom_type to be @p interaction through @p parameters.
     *
     * This function updates or initializes the interaction to use
     * @p parameters, which should be of size ... as defined in
     * .
     */
    void
    declare_interactions(const types::atom_type     i_atom_type,
                         const types::atom_type     j_atom_type,
                         InteractionTypes           interaction,
                         const std::vector<double> &parameters) override;

    /**
     * @copydoc PairCoulWolfManager::energy_and_gradient()
     */
    template <bool ComputeGradient = true>
    inline std::pair<double, double>
    energy_and_gradient(const types::atom_type i_atom_type,
                        const types::atom_type j_atom_type,
                        const double &         squared_distance,
                        const bool             bonded = false) const;

  private:
    /**
     * Born potential computing object.
     */
    PairBornCutManager born_potential;

    /**
     * Class2 potential computing object.
     */
    PairClass2Manager class2_potential;

    /**
     * Coulomb potential computing object using Wolf summation.
     */
    PairCoulWolfManager coul_wolf_potential;
  };



  /*------------------------- Inline functions --------------------------*/

#ifndef DOXYGEN

  template <bool ComputeGradient>
  inline std::pair<double, double>
  PairBornCutClass2CoulWolfManager::energy_and_gradient(
    const types::atom_type i_atom_type,
    const types::atom_type j_atom_type,
    const double &         squared_distance,
    const bool             bonded) const
  {
    const std::pair<double, double> born =
      born_potential.energy_and_gradient<ComputeGradient>(i_atom_type,
                                                          j_atom_type,
                                                          squared_distance,
                                                          bonded);
    const std::pair<double, double> class2 =
      class2_potential.energy_and_gradient<ComputeGradient>(i_atom_type,
                                                            j_atom_type,
                                                            squared_distance,
                                                            bonded);
    const std::pair<double, double> coul =
      coul_wolf_potential.energy_and_gradient<ComputeGradient>(i_atom_type,
                                                               j_atom_type,
                                                               squared_distance,
                                                               bonded);

    return {born.first + class2.first + coul.first,
            born.second + class2.second + coul.second};
  }

#endif /* DOXYGEN */

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_pair_born_cut_class2_coul_wolf_h */
