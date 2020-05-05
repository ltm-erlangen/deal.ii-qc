
#ifndef __dealii_qc_born_cut_h
#define __dealii_qc_born_cut_h

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_base.h>

#include <array>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * Truncated Born-Mayer-Huggins or Tosi-Fumi pair potential,
   * \f[
   *     \phi_{ij} =  A \, \mbox{exp}\bigg[\frac{\sigma-r_{ij}}{\rho} \bigg]
   *               -  \frac{C}{r_{ij}^6}
   *               +  \frac{D}{r_{ij}^8}
   * \f]
   *
   * where \f$\phi_{ij}\f$ is the interacting energy between atom \f$i\f$
   * and atom \f$j\f$ which are \f$r_{ij}\f$ distance apart.
   *
   * The parameters \f$A\f$, \f$\rho\f$, and \f$D\f$ describe components of
   * repulsive interaction and \f$C\f$ describes the attractive interaction.
   * This potential is also called Buckingham potential if \f$\sigma\f$, and
   * \f$D\f$ are set to zero.
   * Only supports InteractionTypes::Born interaction type.
   *
   * @note The above is a modified version of the Born-Mayer potential
   * which has the following form,
   * \f[
   *     \phi_{ij} =  A \, \mbox{exp}\bigg[\frac{\sigma-r_{ij}}{\rho} \bigg].
   * \f]
   */
  class PairBornCutManager : public PairBaseManager
  {
  public:
    /**
     * Constructor that takes in the cutoff radius @p cutoff_radius to be
     * used for computation of energy and it's derivative. The atoms which
     * are farther than @p cutoff_radius do not interact with each other,
     * consequently do not contribute to either energy or it's derivative.
     */
    PairBornCutManager(const double &cutoff_radius);

    /**
     * Declare the type of interaction between the atom types @p i_atom_type
     * and @p j_atom_type to be @p interaction through @p parameters.
     *
     * This function updates or initializes the interaction to use
     * @p parameters, which should be of size five (see #born_parameters).
     */
    void
    declare_interactions(const types::atom_type     i_atom_type,
                         const types::atom_type     j_atom_type,
                         const InteractionTypes     interaction,
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
     * Cutoff radius squared.
     */
    const double cutoff_radius_squared;

    /**
     * A list of potential parameters: \f$A\f$, \f$\frac{1}{\rho}\f$,
     * \f$\sigma\f$, \f$C\f$, and \f$D\f$ (in that order)
     * for each type of Born-Mayer-Huggings interaction.
     */
    std::map<std::pair<types::atom_type, types::atom_type>,
             std::array<double, 5>>
      born_parameters;
  };

  /*------------------------- Inline functions -------------------------*/

#ifndef DOXYGEN

  template <bool ComputeGradient>
  inline std::pair<double, double>
  PairBornCutManager::energy_and_gradient(const types::atom_type i_atom_type,
                                          const types::atom_type j_atom_type,
                                          const double &squared_distance,
                                          const bool    bonded) const
  {
    if (squared_distance > cutoff_radius_squared || bonded)
      return ComputeGradient ?
               std::make_pair(0., 0.) :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

    const std::pair<types::atom_type, types::atom_type> interacting_atom_types =
      get_pair(i_atom_type, j_atom_type);

    const auto &param = born_parameters.find(interacting_atom_types);

    Assert(param != born_parameters.end(),
           dealii::ExcMessage("Born-Mayer-Huggins parameter not set for "
                              "the given interacting atom types"));

    // get Born-Mayer-Huggings parameters
    const double &A      = param->second[0];
    const double &rhoinv = param->second[1];
    const double &sigma  = param->second[2];
    const double &C      = param->second[3];
    const double &D      = param->second[4];

    const double r = std::sqrt(squared_distance);

    const double r2inv = 1.0 / squared_distance;
    const double r6inv = dealii::Utilities::fixed_power<3>(r2inv);
    const double r8inv = r6inv * r2inv;

    const double rexp = std::exp((sigma - r) * rhoinv);

    const double energy = A * rexp - C * r6inv + D * r8inv;

    const double gradient =
      ComputeGradient ?
        -A * rhoinv * rexp + (6 * C * r8inv - 8 * D * r8inv * r2inv) * r :
        std::numeric_limits<double>::signaling_NaN();

    return {energy, gradient};
  }

#endif /* DOXYGEN */

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_born_cut_h */
