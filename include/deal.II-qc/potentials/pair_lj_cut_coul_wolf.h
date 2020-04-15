
#ifndef __dealii_qc_pair_lj_cut_coul_wolf_h
#define __dealii_qc_pair_lj_cut_coul_wolf_h

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_coul_wolf.h>
#include <deal.II-qc/potentials/pair_lj_cut.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * PairLJCutManager with PairCoulWolfManager. Consequently, the energy and
   * gradient values computed using this class is the sum of the two values
   * computed individually.
   */
  class PairLJCutCoulWolfManager : public PairBaseManager
  {
  public:
    /**
     * Constructor that takes in the damping coefficient @p alpha,
     * the cutoff radius @p coul_cutoff_radius to be used for computing Coulomb
     * energy contribution using Wolf summation and the cutoff radius
     * @p lj_cutoff_radius for computing Lennard-Jones contribution.
     * If @p with_tail is `true`, the Lennard-Jones potential has a tail that
     * makes interaction energy converge smoothly to zero as opposed to being
     * abruptly jumping to zero at a separation distance of @p lj_cutoff_radius.
     */
    PairLJCutCoulWolfManager(const double &alpha,
                             const double &coul_cutoff_radius,
                             const double &lj_cutoff_radius,
                             const bool    with_tail = false);

    /**
     * See PairBaseManager::set_charges().
     */
    void
    set_charges(std::shared_ptr<std::vector<types::charge>> &charges_);

    /**
     * Declare the type of interaction between the atom types @p i_atom_type
     * and @p j_atom_type to be @p interaction through @p parameters.
     *
     * This function updates or initializes the interaction to use
     * @p parameters, which should be of size two with the first element
     * being \f$\epsilon\f$ and second being \f$r_m\f$ as defined in
     * PairLJCutManager.
     */
    void
    declare_interactions(const types::atom_type     i_atom_type,
                         const types::atom_type     j_atom_type,
                         InteractionTypes           interaction,
                         const std::vector<double> &parameters);

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
     * LJ potential computing object.
     */
    PairLJCutManager lj_potential;

    /**
     * Coulomb potential computing object using Wolf summation.
     */
    PairCoulWolfManager coul_wolf_potential;
  };



  /*----------------------- Inline functions
   * ----------------------------------*/

#ifndef DOXYGEN

  template <bool ComputeGradient>
  inline std::pair<double, double>
  PairLJCutCoulWolfManager::energy_and_gradient(
    const types::atom_type i_atom_type,
    const types::atom_type j_atom_type,
    const double &         squared_distance,
    const bool             bonded) const
  {
    if (bonded)
      return {0., 0.};

    const std::pair<double, double> lj =
      lj_potential.energy_and_gradient<ComputeGradient>(i_atom_type,
                                                        j_atom_type,
                                                        squared_distance,
                                                        false);
    const std::pair<double, double> coul =
      coul_wolf_potential.energy_and_gradient<ComputeGradient>(i_atom_type,
                                                               j_atom_type,
                                                               squared_distance,
                                                               false);

    return {lj.first + coul.first, lj.second + coul.second};
  }

#endif /* DOXYGEN */

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_pair_lj_cut_coul_wolf_h */
