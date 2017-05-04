
#include <cmath>

#include <dealiiqc/potentials/pair_coul_wolf.h>

namespace dealiiqc
{
  using namespace dealii;

  namespace Potential
  {
    void
    PairBaseManager::set_charges (std::shared_ptr<std::vector<types::charge>> &charges_)
    {
      // non-virtual function
      // Update the shared pointer to point to the vector of charges
      // (whose size should be of size equal to the number of different
      //  atom types)
      charges = charges_;
    }

    PairCoulWolfManager::PairCoulWolfManager ( const double &alpha,
                                               const double &cutoff_radius)
      :
      alpha(alpha),
      cutoff_radius(cutoff_radius),
      energy_shift(std::erfc(alpha *cutoff_radius)/cutoff_radius),
      cutoff_radius_inverse(1./cutoff_radius),
      compound_exp_value(std::exp(-alpha *alpha *cutoff_radius *cutoff_radius))
    {}


    void
    PairCoulWolfManager::declare_interactions ( const types::atom_type i_atom_type,
                                                const types::atom_type j_atom_type,
                                                InteractionTypes interaction,
                                                const std::vector<double> &parameters)
    {
      Assert( interaction==InteractionTypes::Coul_Wolf,
              ExcMessage("Invalid InteractionTypes specified"));

      Assert( i_atom_type < charges->size() && j_atom_type < charges->size(),
              ExcMessage("Either the list of charges is initialized incorrectly"
                         "Or atom type argument passed is incorrect"));

      Assert( parameters.size() == 0,
              ExcMessage("This class does not accept any parameters."));
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

      Assert (charges, ExcInternalError());

      // TODO: Need to setup units
      // The multiplying factor qqrd2e = 14.399645 yields energy in eV
      // and force in eV/Angstrom units
      const double qqrd2e = 14.399645;
      const double distance = std::sqrt(squared_distance);

      Assert( i_atom_type < charges->size() && i_atom_type < charges->size(),
              ExcMessage("The function is called with a value of atom type "
                         "larger than the size of PairCoulWolf::charges."
                         "Please ensure that the PairCoulWolf::charges is "
                         "initialized accurately."));

      const double qiqj = (double) PairBaseManager::charges->operator[](i_atom_type)*
                          PairBaseManager::charges->operator[](j_atom_type);
      const double distance_inverse = 1.0/distance;
      const double erfc_a_distance = std::erfc(alpha*distance) * distance_inverse;

      const double energy = qiqj * ( erfc_a_distance - energy_shift ) * qqrd2e;

      const double force = ComputeScalarForce
                           ?
                           qqrd2e * qiqj *
                           ( distance_inverse *
                             (
                               erfc_a_distance + alpha*M_2_SQRTPI *
                               std::exp(-alpha*alpha*squared_distance)
                             )
                             -
                             cutoff_radius_inverse *
                             (
                               energy_shift + alpha * M_2_SQRTPI *
                               compound_exp_value
                             )
                           )
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
