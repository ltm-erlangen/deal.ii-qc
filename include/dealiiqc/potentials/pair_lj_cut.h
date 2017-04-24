
#ifndef __dealii_qc_pair_lj_cut_h
#define __dealii_qc_pair_lj_cut_h

#include <dealiiqc/utilities.h>

namespace dealiiqc
{

  namespace Potential
  {
    // TODO: Move this to potentials.h
    /**
     * An enumeration of all the pair potentials implemented in
     * Potentials namespace.
     */
    enum InteractionTypes
    {
      /**
       * @p Lennard-Jones indicates that the pair potential manager
       * PairLJCutManager is implemented in the Potentials namespace.
       */
      LJ = 0
    };


    /**
     * Truncated Lennard-Jones pair potential.
     * Only support InteractionTypes::LJ interaction type.
     *
     * \f[
     *     \phi_{ij} =  \epsilon \left\[    (\frac{r_m}{r_{ij}})^12
     *                                   - 2(\frac{r_m}{r_{ij}})^6  \right\]
     * \f]
     *
     * where \f$\phi_{ij}$\f is the interactiong energy between atom \f$i$\f
     * and atom \f$j$\f which are \f$r_{ij}$\f distance apart.
     *
     * The parameter \f$\epsilon$\f is the depth of the potential well and
     * \f$r_m$\f is the distance between atoms such that the interaction energy
     * between the two atoms is minimum (equal to \f$\epsilon$\f).
     *
     * @note: The above is a modified version of classical Lennard-Jones
     * potential which has the following form,
     * \f[
     *     \phi_{ij} =  4 \epsilon \left\[   (\frac{\sigma}{r_{ij}})^12
     *                                     - (\frac{\sigma}{r_{ij}})^6  \right\]
     * \f]
     *
     * where \f$r_m$\f is \f$2^{1/6}\sigma$\f.
     */
    class PairLJCutManager
    {

    public:

      /**
       * Constructor that takes in the cutoff radius @p cutoff_radius to be
       * used for computation of energy and it's derivative. The atoms which
       * are farther than @p cutoff_radius do not interact with each other,
       * consequently do not contribute to either energy or it's derivative.
       */
      PairLJCutManager( const double &cutoff_radius);

      /**
       * Declare the type of interaction between the atom types @p i_atom_type
       * and @p j_atom_type to be @p interaction through @p parameters.
       *
       * This function updates or initializes the interaction to use
       * @p parameters, which should be of size two with the first element
       * being \f$\epsilon$\f and second being \f$r_m$\f as defined in
       * PairLJCutManager.
       */
      void declare_interactions ( const types::atom_type i_atom_type,
                                  const types::atom_type j_atom_type,
                                  InteractionTypes interaction,
                                  std::vector<double> &parameters);


      /**
       * Returns a pair of computed value of energy and scalar force between
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
       * Cutoff radius.
       */
      const double cutoff_radius;

      /**
       * A list of minimum LJ energy values (depths of the potential wells)
       * due to interaction between different atom types.
       */
      std::map<std::pair<types::atom_type, types::atom_type>, double> epsilon;

      /**
       * A list of distances at which LJ energy values due to interaction
       * between different atom types reaches a minimum.
       */
      std::map<std::pair<types::atom_type, types::atom_type>, double> r_m;

    };

  }

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_lj_cut_h */
