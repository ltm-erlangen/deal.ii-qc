
#include <cmath>

#include <dealiiqc/potentials/pair_coul_wolf.h>

namespace dealiiqc
{
  using namespace dealii;

  namespace Potential
  {

    PairCoulWolfManager::PairCoulWolfManager ( const double &alpha,
                                               const double &cutoff_radius,
                                               const std::vector<types::charge> &charges)
      :
      alpha(alpha),
      cutoff_radius(cutoff_radius),
      charges(charges)
    {}

    void
    PairCoulWolfManager::declare_interactions ( const types::atom_type i_atom_type,
                                                const types::atom_type j_atom_type,
                                                InteractionTypes interaction,
                                                const std::vector<double> &parameters)
    {
      Assert( interaction==InteractionTypes::Coul_Wolf,
              ExcMessage("Invalid InteractionTypes specified"));

      Assert( i_atom_type < charges.size() || j_atom_type < charges.size(),
              ExcMessage("Either the list of charges is initialized incorrectly"
                         "Or atom type argument passed is incorrect"));

    }

    template<bool ComputeScalarForce>
    std::pair<double, double>
    PairCoulWolfManager::energy_and_scalar_force ( const types::atom_type i_atom_type,
                                                   const types::atom_type j_atom_type,
                                                   const double &squared_distance) const
    {
      if ( squared_distance > cutoff_radius*cutoff_radius )
        return ComputeScalarForce
               ?
               std::make_pair(0.,0.)
               :
               std::make_pair(0., std::numeric_limits<double>::signaling_NaN());

      // TODO: Need to setup units
      // The multiplying factor qqrd2e = 14.399645 yields energy in eV
      // and force in eV/Angstrom units
      const double qqrd2e = 14.399645;
      const double distance = std::sqrt(squared_distance);

      const std::pair<types::atom_type, types::atom_type>
      interacting_atom_types = get_pair( i_atom_type, j_atom_type);

      const double qiqj = (double) charges[i_atom_type]*charges[j_atom_type];
      const double energy_shift = std::erfc(alpha*cutoff_radius)/cutoff_radius;

      const double energy = qiqj * ( std::erfc(alpha*distance)/distance -
                                     std::erfc(alpha*cutoff_radius)/cutoff_radius ) * qqrd2e;

      const double force = ComputeScalarForce
                           ?
                           (
                             std::erfc(alpha*distance)/squared_distance
                             +
                             alpha*M_2_SQRTPI * std::exp(-alpha*alpha*distance*distance)/distance
                             -
                             (
                               energy_shift + alpha * M_2_SQRTPI
                               * std::exp(-alpha*alpha*cutoff_radius*cutoff_radius)
                             )/ cutoff_radius
                           ) * qiqj * qqrd2e
                           :
                           std::numeric_limits<double>::signaling_NaN();

      return std::make_pair(energy,force);
    }

    template
    std::pair<double, double>
    PairCoulWolfManager::energy_and_scalar_force <true> ( const types::atom_type i_atom_type,
                                                          const types::atom_type j_atom_type,
                                                          const double &squared_distance) const;

    template
    std::pair<double, double>
    PairCoulWolfManager::energy_and_scalar_force <false> ( const types::atom_type i_atom_type,
                                                           const types::atom_type j_atom_type,
                                                           const double &squared_distance) const;
  }

}
