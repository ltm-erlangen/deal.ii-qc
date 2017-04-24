
#include <limits>

#include <deal.II/base/utilities.h>

#include <dealiiqc/potentials/pair_lj_cut.h>

namespace dealiiqc
{
  using namespace dealii;

  namespace Potential
  {

    // TODO: Move this to potentials.h
    /**
     * Return a pair of atom type such that the first element is less than
     * or equal to that of the second element given the two atom types
     * @p i_atom_type and @p j_atom_type.
     */
    inline
    std::pair<types::atom_type, types::atom_type>
    get_pair (const types::atom_type i_atom_type,
              const types::atom_type j_atom_type)
    {
      return ( i_atom_type <= j_atom_type)
             ?
             std::make_pair( i_atom_type, j_atom_type)
             :
             std::make_pair( j_atom_type, i_atom_type);
    }


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

      epsilon.insert( std::make_pair( get_pair(i_atom_type, j_atom_type),
                                      parameters[0]) );

      r_m.insert( std::make_pair( get_pair(i_atom_type, j_atom_type),
                                  parameters[1]) );
    }

    template<bool ComputeScalarForce>
    std::pair<double, double>
    PairLJCutManager::energy_and_scalar_force ( const types::atom_type i_atom_type,
                                                const types::atom_type j_atom_type,
                                                const double &squared_distance) const
    {

      if ( squared_distance > cutoff_radius*cutoff_radius )
        return ComputeScalarForce
               ?
               std::make_pair(0.,0.)
               :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

      const std::pair<types::atom_type, types::atom_type>
      interacting_atom_types = get_pair( i_atom_type, j_atom_type);

      Assert( epsilon.count(interacting_atom_types),
              ExcMessage("LJ parameter not set for the given interacting atom types"));

      // get LJ parameters
      // TODO: Move rm6 to a seperate map
      //       so that rm6 is precomputed as opposed to computing each time this
      //       function is called
      const double rm6  = dealii::Utilities::fixed_power<6>(r_m.find(interacting_atom_types)->second);
      const double eps  = epsilon.find(interacting_atom_types)->second;

      const double rm_by_r6 = rm6 / dealii::Utilities::fixed_power<3>(squared_distance);

      const double energy = eps * rm_by_r6 * ( rm_by_r6 - 2.);
      const double force  = ComputeScalarForce
                            ?
                            -12. * eps * rm_by_r6 * ( 1. - rm_by_r6) / sqrt(squared_distance)
                            :
                            std::numeric_limits<double>::signaling_NaN();

      return std::make_pair(energy, force);
    }

    template
    std::pair<double, double>
    PairLJCutManager::energy_and_scalar_force<true> ( const types::atom_type i_atom_type,
                                                      const types::atom_type j_atom_type,
                                                      const double &squared_distance) const;

    template
    std::pair<double, double>
    PairLJCutManager::energy_and_scalar_force<false> ( const types::atom_type i_atom_type,
                                                       const types::atom_type j_atom_type,
                                                       const double &squared_distance) const;

  }

} /* namespace dealiiqc */
