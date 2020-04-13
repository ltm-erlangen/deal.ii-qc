
#ifndef __dealii_qc_pair_class2_h
#define __dealii_qc_pair_class2_h

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_base.h>

#include <array>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{
  /**
   * The Class2 (bond) pair potential.
   * Only supports InteractionType::Class2 interaction type.
   *
   * \f[
   *     \phi_{ij} =  k_2 \left[ r_{ij} - r_{m} \right]^2 +
   *                  k_3 \left[ r_{ij} - r_{m} \right]^3 +
   *                  k_4 \left[ r_{ij} - r_{m} \right]^4.
   * \f]
   *
   * where \f$\phi_{ij}\f$ is the interacting energy between atom \f$i\f$
   * and atom \f$j\f$ which are \f$r_{ij}\f$ distance apart.
   *
   * The parameters \f$k_2, k_3,\f$ and \f$k_4\f$ are bond parameters and
   * \f$r_m\f$ is the distance between atoms such that the interaction energy
   * between the two atoms is minimum (equal to zero).
   */
  class PairClass2Manager : public PairBaseManager
  {
  public:
    /**
     * Constructor.
     */
    PairClass2Manager();

    /**
     * Return whether the potential is a bond potential or
     * has an augmented bond potential.
     */
    virtual bool
    is_or_has_bond_style() const override;

    /**
     * Declare the type of @p interaction between the atom types @p i_atom_type
     * and @p j_atom_type through @p parameters.
     *
     * This function updates or initializes the interaction to use
     * @p parameters, which should be of size two with the first element
     * being \f$\epsilon\f$ and second being \f$r_m\f$ as defined in
     * PairLJCutManager.
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
                        const double &         squared_distance) const;

  private:
    /**
     * A list of four parameters  \f$r_m, k_2, k_3,\f$ and \f$k_4\f$
     * are specified for each interaction.
     */
    std::map<std::pair<types::atom_type, types::atom_type>,
             std::array<double, 4>>
      class2_parameters;
  };

  /*------------------------ Inline functions -------------------------------*/

#ifndef DOXYGEN

  template <bool ComputeGradient>
  inline std::pair<double, double>
  PairClass2Manager::energy_and_gradient(const types::atom_type i_atom_type,
                                         const types::atom_type j_atom_type,
                                         const double &squared_distance) const
  {
    const std::pair<types::atom_type, types::atom_type> interacting_atom_types =
      get_pair(i_atom_type, j_atom_type);

    const auto &param = class2_parameters.find(interacting_atom_types);

    if (param == class2_parameters.end())
      return ComputeGradient ?
               std::make_pair(0., 0.) :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

    // get Class2 parameters
    const double &rm = param->second[0];
    const double &k2 = param->second[1];
    const double &k3 = param->second[2];
    const double &k4 = param->second[3];

    const double d  = std::sqrt(squared_distance) - rm;
    const double d2 = d * d;
    const double d3 = d * d2;

    const double energy = k2 * d2 + k3 * d3 + k4 * d3 * d;

    const double gradient = ComputeGradient ?
                              2 * k2 * d + 3 * k3 * d2 + 4 * k4 * d3 :
                              std::numeric_limits<double>::signaling_NaN();

    return {energy, gradient};
  }

#endif /* DOXYGEN */

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_pair_class2_h */
