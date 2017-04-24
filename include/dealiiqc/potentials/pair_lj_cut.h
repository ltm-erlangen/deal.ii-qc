
#ifndef __dealii_qc_pair_lj_cut_h
#define __dealii_qc_pair_lj_cut_h

#include <dealiiqc/utilities.h>

namespace dealiiqc
{

  namespace Potential
  {
    enum InteractionTypes
    {
      LJ = 0
    };


    /**
     * Truncated Lennard-Jones pair potential. Only support LJ interaction type.
     *
     * \f[
     *     \phi_{ij} =  \epsilon \left\[    (\frac{r_m}{r_{ij}})^12
     *                                   - 2(\frac{r_m}{r_{ij}})^6  \right\]
     * \f]
     */
    class PairLJCutManager
    {

    public:

      /**
       * Constructor.
       */
      PairLJCutManager( const double &cutoff_radius);

      /**
       * Destructor.
       */
      virtual ~PairLJCutManager();

      /**
       * Declare interacting pair of atom types.
       *
       * This function updates or initializes interacting_atom_types
       * and parameters used for
       */
      void declare_interactions ( const types::atom_type i_atom_type,
                                  const types::atom_type j_atom_type,
                                  InteractionTypes interaction,
                                  std::vector<double> &parameters);


      /**
       * Returns a pair of computed value of energy and scalar force between
       * two atoms with atom_type @p i_atom_type and atom_type @p j_atom_type
       * that are a distance of square root of @p squared_distance apart.
       * The template parameter indicates whether to skip the additional
       * computation of scalar force; this is in the case when only the
       * value of the energy is intended to be queried.
       *
       * A typical energy minimization process might need the value of
       * energy much more often than the value of force. Therefore,
       * this function can be called by passing @p true as template
       * parameter to query computed value of the energy.
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
