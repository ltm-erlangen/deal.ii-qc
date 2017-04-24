
#include <deal.II/base/utilities.h>
#include <limits>

#include <dealiiqc/potentials/pair_lj_cut.h>

namespace dealiiqc
{

  namespace Potential
  {

    PairLJCutManager::PairLJCutManager( const double &cutoff_radius)
      :
      cutoff_radius(cutoff_radius)
    {}

    void
    PairLJCutManager::declare_interactions ( const types::atom_type i_atom_type,
                                             const types::atom_type j_atom_type,
                                             InteractionTypes interaction,
                                             std::vector<double> &parameters)
    {
      Assert( interaction==InteractionTypes::LJ,
              ExcMessage("Invalid InteractionTypes specified"));

      if ( i_atom_type <= j_atom_type )
        {
          epsilon.insert( std::make_pair( std::make_pair( i_atom_type, j_atom_type),
                                          parameters[0]) );
          r_m.insert( std::make_pair( std::make_pair( i_atom_type, j_atom_type),
                                      parameters[1]) );
        }
      else
        {
          epsilon.insert( std::make_pair( std::make_pair( j_atom_type, i_atom_type),
                                          parameters[0]) );
          r_m.insert( std::make_pair( std::make_pair( j_atom_type, i_atom_type),
                                      parameters[1]) );
        }

    }

    template<bool ComputeScalarForce=true>
    std::pair<double, double>
    PairLJCutManager::energy_and_scalar_force ( const types::atom_type i_atom_type,
                                                const types::atom_type j_atom_type,
                                                const double &squared_distance) const
    {
      std::pair<types::atom_type, types::atom_type> interacting_atom_types =
        (i_atom_type <= j_atom_type)
        ?
        std::make_pair(i_atom_type, j_atom_type)
        :
        std::make_pair(j_atom_type, i_atom_type);

      Assert( epsilon.count(interacting_atom_types),
              "LJ parameter not set for the given interacting atom types");

      // get LJ parameters
      // TODO: Move rm6 to a seperate map
      //       so that rm6 is precomputed as opposed to computing each time this
      //       function is called
      const double rm6  = dealii::Utilities::fixed_power<6>(r_m[interacting_atom_types]);
      const double eps  = epsilon[interacting_atom_types];


      const double rm_by_r6 = rm6 / dealii::Utilities::fixed_power<3>(squared_distance);

      const double energy = epsilon * rm_by_r6 * ( rm_by_r6 - 2. );
      const double force  = std::numeric_limits<double>::signaling_NaN();

      if (ComputeScalarForce)
        {
          const double distance_inv = 1. / sqrt(squared_distance);
          force = -12. * energy * distance_inv;
        }

      return std::make_pair(energy, force);

    }

  }

} /* namespace dealiiqc */
