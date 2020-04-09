
#ifndef __dealii_qc_pair_coul_wolf_h
#define __dealii_qc_pair_coul_wolf_h

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * Coulomb pair potential computed using Wolf summation method as
   *
   * \f[
   *     \phi_{ij} =  \frac{q_i \, q_j \, \mbox{erfc}(\alpha r_{ij})}{r_{ij}}
   *                  -
   *                  \lim_{r \to r_c}
   *                  \frac{q_i \, q_j \, \mbox{erfc}(\alpha r)}{r}
   * \f]
   *
   * where \f$\phi_{ij}\f$ is the Coulomb interaction energy between atom
   * \f$i\f$ and atom \f$j\f$ with charges \f$q_i\f$ and \f$q_j\f$
   * which are \f$r_{ij}\f$ distance apart. The parameter \f$\alpha\f$ is
   * the damping coefficient and \f$r_c\f$ is the cutoff radius.
   * This pair potential only supports InteractionTypes::Coul_Wolf interaction
   * type.
   *
   * @note The contribution of self energy \f$E^s\f$ is not computed
   * within PairCoulWolfManager::energy_and_gradient(). Therefore, the
   * value of the energy is shifted by a value of \f$E^s\f$ given by
   *
   * \f[
   *      E^s =  - \left[ \frac{\mbox{erfc}(\alpha r_{c})}{2 r_{c}}
   *                 +    \frac{\alpha}{\sqrt{\pi}} \right] \sum_i^N q_i^2
   * \f]
   */
  class PairCoulWolfManager : public PairBaseManager
  {
  public:
    /**
     * Constructor that takes in the damping coefficient @p alpha,
     * the cutoff radius @p cutoff_radius to be used for computation of
     * Coulomb energy and it's derivative using Wolf summation.
     * The atoms which are farther than @p cutoff_radius do not
     * interact with each other, consequently do not contribute to either
     * energy or its derivative.
     */
    PairCoulWolfManager(const double &alpha, const double &cutoff_radius);

    /**
     * Declare the type of interaction between the atom types @p i_atom_type
     * and @p j_atom_type to be @p interaction. @p parameters is not used by
     * this class and is present here only for consistency with the other
     * potentials.
     */
    void
    declare_interactions(
      const types::atom_type     i_atom_type,
      const types::atom_type     j_atom_type,
      InteractionTypes           interaction,
      const std::vector<double> &parameters = std::vector<double>());

    /**
     * Returns a pair of computed values of energy and its gradient between
     * two atoms of type @p i_atom_type and type @p j_atom_type
     * that are a distance of square root of @p squared_distance apart.
     * The first value in the returned pair is energy whereas the second
     * is its partial derivative given as \f$
     * \frac{\partial \phi_{ij}}{\partial r_{ij}} \f$.
     *
     * The template parameter indicates whether to skip the additional
     * computation of gradient; this is in the case when only the
     * value of the energy is intended to be queried.
     *
     * @note A typical energy minimization process might need the value of
     * energy much more often than the value of force. Therefore,
     * this function can be called by passing @p false as template
     * parameter to query only the computation of the energy.
     */
    template <bool ComputeGradient = true>
    inline std::pair<double, double>
    energy_and_gradient(const types::atom_type i_atom_type,
                        const types::atom_type j_atom_type,
                        const double &         squared_distance) const;

  private:
    /**
     * Damping coefficient.
     */
    const double alpha;

    /**
     * Cutoff radius.
     */
    const double cutoff_radius;

    /**
     * Cutoff radius squared.
     */
    const double cutoff_radius_squared;

    /**
     * A const member whose value is required during energy computations.
     */
    const double energy_shift;

    /**
     * A const member whose value is required during energy computations.
     */
    const double cutoff_radius_inverse;

    /**
     * A const member whose value is required during energy computations.
     */
    const double compound_exp_value;
  };



  /*----------------------- Inline functions
   * ----------------------------------*/

#ifndef DOXYGEN

  template <bool ComputeGradient>
  inline std::pair<double, double>
  PairCoulWolfManager::energy_and_gradient(const types::atom_type i_atom_type,
                                           const types::atom_type j_atom_type,
                                           const double &squared_distance) const
  {
    if (squared_distance > cutoff_radius_squared)
      return ComputeGradient ?
               std::make_pair(0., 0.) :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

    Assert(charges, dealii::ExcInternalError());

    // TODO: Need to setup units
    // The multiplying factor qqrd2e = 14.399645 yields energy in eV
    // and force in eV/Angstrom units
    const double qqrd2e   = 14.399645;
    const double distance = std::sqrt(squared_distance);

    Assert(i_atom_type < charges->size() && i_atom_type < charges->size(),
           dealii::ExcMessage("The function is called with a value of "
                              "atom type larger than the size of "
                              "PairCoulWolf::charges."
                              "Please ensure that the PairCoulWolf::charges "
                              "is initialized accurately."));

    const double qiqj =
      (double)(*charges)[i_atom_type] * (*charges)[j_atom_type];
    const double distance_inverse = 1.0 / distance;
    const double erfc_a_distance =
      std::erfc(alpha * distance) * distance_inverse;

    const double energy = qiqj * (erfc_a_distance - energy_shift) * qqrd2e;

    const double gradient =
      ComputeGradient ?
        qqrd2e * qiqj *
          (distance_inverse *
             (erfc_a_distance + alpha * M_2_SQRTPI *
                                  std::exp(-alpha * alpha * squared_distance)) -
           cutoff_radius_inverse *
             (energy_shift + alpha * M_2_SQRTPI * compound_exp_value)) :
        std::numeric_limits<double>::signaling_NaN();

    return std::make_pair(energy, gradient);
  }

#endif /* DOXYGEN */

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_pair_coul_wolf_h */
