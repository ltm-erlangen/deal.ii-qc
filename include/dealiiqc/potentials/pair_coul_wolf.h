
#ifndef __dealii_qc_pair_coul_wolf_h
#define __dealii_qc_pair_coul_wolf_h

#include <dealiiqc/potentials/potentials.h>
#include <dealiiqc/utilities.h>

namespace dealiiqc
{

  namespace Potential
  {

    /**
     * Coulomb pair potential computed using Wolf summation method.
     * Only supports InteractionTypes::Coul_Wolf interaction type.
     *
     * \f[
     *     \phi_{ij} =  \frac{q_iq_j \mbox{erfc}(\alpha r_{ij})}{r_{ij}}
     *               -  \frac{q_iq_j \mbox{erfc}(\alpha r_{c})}{r_{c}}
     * \f]
     *
     * where \f$\phi_{ij}$\f is the Coulomb interaction energy between atom
     * \f$i$\f and atom \f$j$\f with charges \f$q_i$\f and \f$q_j$\f
     * which are \f$r_{ij}$\f distance apart. The parameter \f$\alpha$\f is
     * the damping coefficient and \f$r_c$\f is the cutoff radius.
     *
     * //TODO: Write if energy_self is computed by this class.
     */
    class PairCoulWolfManager
    {

    public:

      /**
       * Constructor that takes in the damping coefficient @p alpha,
       * the cutoff radius @p cutoff_radius, and a list of charges @p charges
       * (whose size should be of size equal to the number of different
       * atom types) to be used for computation of accumulation of Coulomb
       * energy and it's derivative using wolf summation.
       * The atoms which are farther than @p cutoff_radius do not
       * interact with each other, consequently do not contribute to either
       * energy or it's derivative.
       */
      PairCoulWolfManager( const double &alpha,
                           const double &cutoff_radius,
                           const std::vector<types::charge> &charges);

      /**
       * Declare the type of interaction between the atom types @p i_atom_type
       * and @p j_atom_type to be @p interaction.
       */
      void declare_interactions ( const types::atom_type i_atom_type,
                                  const types::atom_type j_atom_type,
                                  InteractionTypes interaction,
                                  const std::vector<double> &parameters=std::vector<double>());

      /**
       * Returns a pair of computed values of energy and scalar force between
       * two atoms with atom type @p i_atom_type and atom_type @p j_atom_type
       * that are a distance of square root of @p squared_distance apart.
       * The first value in the returned pair is energy whereas the second
       * is its (scalar) derivative, i.e. scalar force.
       * The template parameter indicates whether to skip the additional
       * computation of scalar force; this is in the case when only the
       * value of the energy is intended to be queried.
       *
       * @note: A typical energy minimization process might need the value of
       * energy much more often than the value of force. Therefore,
       * this function can be called by passing @p false as template
       * parameter to query only the computation of the energy.
       */
      template<bool ComputeScalarForce=true>
      inline
      std::pair<double, double>
      energy_and_scalar_force ( const types::atom_type i_atom_type,
                                const types::atom_type j_atom_type,
                                const double &squared_distance) const;

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
       * A list of charges \f$q_i$\f of the different the atom types.
       */
      const std::vector<types::charge> &charges;

    };

  } // namespace Potential

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_coul_wolf_h */
