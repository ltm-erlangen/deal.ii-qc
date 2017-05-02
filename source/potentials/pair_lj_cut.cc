
#include <limits>

#include <deal.II/base/utilities.h>

#include <dealiiqc/potentials/pair_lj_cut.h>

namespace dealiiqc
{
  using namespace dealii;

  namespace Potential
  {

    PairLJCutManager::PairLJCutManager( const double &cutoff_radius)
      :
      cutoff_radius_squared(cutoff_radius *cutoff_radius)
    {}

    void
    PairLJCutManager::declare_interactions ( const types::atom_type i_atom_type,
                                             const types::atom_type j_atom_type,
                                             const InteractionTypes interaction,
                                             const std::vector<double> &parameters)
    {
      Assert( interaction==InteractionTypes::LJ,
              ExcMessage("Invalid InteractionTypes specified"));
      Assert (parameters.size() == 2,
              ExcMessage("Invalid parameters list"));
      Assert (parameters[0] > 0.,
              ExcMessage("Invalid epsilon value specified for LJ pair potential"));
      Assert (parameters[1] > 0.,
              ExcMessage("Invalid r_m value specified for LJ pair potential"));

      const std::array<double, 2> params = {{parameters[0],dealii::Utilities::fixed_power<6>(parameters[1])}};
      lj_parameters.insert( std::make_pair( get_pair(i_atom_type, j_atom_type),
                                            params) );

    }

    template<bool ComputeScalarForce>
    std::pair<double, double>
    PairLJCutManager::energy_and_scalar_force ( const types::atom_type i_atom_type,
                                                const types::atom_type j_atom_type,
                                                const double &squared_distance) const
    {

      if ( squared_distance > cutoff_radius_squared)
        return ComputeScalarForce
               ?
               std::make_pair(0.,0.)
               :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

      const std::pair<types::atom_type, types::atom_type>
      interacting_atom_types = get_pair( i_atom_type, j_atom_type);

      const auto &param = lj_parameters.find(interacting_atom_types);

      Assert( param != lj_parameters.end(),
              ExcMessage("LJ parameter not set for the given interacting atom types"));

      // get LJ parameters
      const double &eps = param->second[0];
      const double &rm6 = param->second[1];

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

  } // namespace Potential

} /* namespace dealiiqc */
