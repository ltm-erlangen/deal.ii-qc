
#include <cmath>

#include <deal.II-qc/potentials/pair_coul_wolf.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{


  PairCoulWolfManager::PairCoulWolfManager (const double &alpha,
                                            const double &cutoff_radius)
    :
    alpha(alpha),
    cutoff_radius(cutoff_radius),
    energy_shift(std::erfc(alpha *cutoff_radius)/cutoff_radius),
    cutoff_radius_inverse(1./cutoff_radius),
    compound_exp_value(std::exp(-alpha *alpha *cutoff_radius *cutoff_radius))
  {}



  void
  PairCoulWolfManager::declare_interactions (const types::atom_type i_atom_type,
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

} // namespace Potential

DEAL_II_QC_NAMESPACE_CLOSE
