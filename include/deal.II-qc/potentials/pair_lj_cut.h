
#ifndef __dealii_qc_pair_lj_cut_h
#define __dealii_qc_pair_lj_cut_h

#include <array>

#include <deal.II/base/exceptions.h>

#include <deal.II-qc/potentials/pair_base.h>

namespace dealiiqc
{

  namespace Potential
  {

    /**
     * Truncated Lennard-Jones pair potential.
     * Only supports InteractionTypes::LJ interaction type.
     *
     * \f[
     *     \phi_{ij} =  \epsilon \left[ \left(\frac{r_m}{r_{ij}}\right)^{12}
     *                               - 2\left(\frac{r_m}{r_{ij}}\right)^6  \right]
     * \f]
     *
     * where \f$\phi_{ij}\f$ is the interacting energy between atom \f$i\f$
     * and atom \f$j\f$ which are \f$r_{ij}\f$ distance apart.
     *
     * The parameter \f$\epsilon\f$ is the depth of the potential well and
     * \f$r_m\f$ is the distance between atoms such that the interaction energy
     * between the two atoms is minimum (equal to \f$\epsilon\f$).
     *
     * @note The above is a modified version of classical Lennard-Jones
     * potential which has the following form,
     * \f[
     *     \phi_{ij} =  4 \epsilon \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12}
     *                                   -\left(\frac{\sigma}{r_{ij}}\right)^6  \right]
     * \f]
     *
     * where \f$r_m=2^{1/6}\sigma\f$.
     */
    class PairLJCutManager : public PairBaseManager
    {

    public:

      /**
       * Constructor that takes in the cutoff radius @p cutoff_radius to be
       * used for computation of energy and it's derivative. The atoms which
       * are farther than @p cutoff_radius do not interact with each other,
       * consequently do not contribute to either energy or it's derivative.
       */
      PairLJCutManager (const double &cutoff_radius);

      /**
       * Declare the type of interaction between the atom types @p i_atom_type
       * and @p j_atom_type to be @p interaction through @p parameters.
       *
       * This function updates or initializes the interaction to use
       * @p parameters, which should be of size two with the first element
       * being \f$\epsilon\f$ and second being \f$r_m\f$ as defined in
       * PairLJCutManager.
       */
      void declare_interactions (const types::atom_type i_atom_type,
                                 const types::atom_type j_atom_type,
                                 const InteractionTypes interaction,
                                 const std::vector<double> &parameters);


      /**
       * @copydoc PairCoulWolfManager::energy_and_gradient()
       */
      template<bool ComputeGradient=true>
      inline
      std::pair<double, double>
      energy_and_gradient (const types::atom_type i_atom_type,
                           const types::atom_type j_atom_type,
                           const double &squared_distance) const;

    private:

      /**
       * Cutoff radius squared.
       */
      const double cutoff_radius_squared;

      /**
       * A list of two parameters corresponding to
       * - minimum LJ energy values (depths of the potential wells), and
       * - distances raised to the power six at which LJ energy values
       * reaches a minimum due to interaction between different atom types.
       */
      std::map<std::pair<types::atom_type, types::atom_type>, std::array<double,2> > lj_parameters;

    };

    /*----------------------- Inline functions ----------------------------------*/

#ifndef DOXYGEN

    template<bool ComputeGradient>
    inline
    std::pair<double, double>
    PairLJCutManager::energy_and_gradient (const types::atom_type i_atom_type,
                                           const types::atom_type j_atom_type,
                                           const double &squared_distance) const
    {

      if ( squared_distance > cutoff_radius_squared)
        return ComputeGradient
               ?
               std::make_pair(0.,0.)
               :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

      const std::pair<types::atom_type, types::atom_type>
      interacting_atom_types = get_pair( i_atom_type, j_atom_type);

      const auto &param = lj_parameters.find(interacting_atom_types);

      Assert( param != lj_parameters.end(),
              dealii::ExcMessage("LJ parameter not set for the given interacting atom types"));

      // get LJ parameters
      const double &eps = param->second[0];
      const double &rm6 = param->second[1];

      const double rm_by_r6 = rm6 / dealii::Utilities::fixed_power<3>(squared_distance);

      const double energy = eps * rm_by_r6 * ( rm_by_r6 - 2.);
      const double force  = ComputeGradient
                            ?
                            -12. * eps * rm_by_r6 * ( 1. - rm_by_r6) / sqrt(squared_distance)
                            :
                            std::numeric_limits<double>::signaling_NaN();

      return std::make_pair(energy, force);
    }

#endif /* DOXYGEN */

  } // namespace Potential

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_lj_cut_h */