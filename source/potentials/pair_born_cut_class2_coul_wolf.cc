
#include <deal.II/base/utilities.h>

#include <deal.II-qc/potentials/pair_born_cut_class2_coul_wolf.h>

#include <limits>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{
  PairBornCutClass2CoulWolfManager::PairBornCutClass2CoulWolfManager(
    const double &alpha,
    const double &coul_cutoff_radius,
    const double &born_cutoff_radius,
    const double &factor_coul)
    : born_potential(born_cutoff_radius)
    , class2_potential()
    , coul_wolf_potential(alpha, coul_cutoff_radius, factor_coul)
  {}



  void
  PairBornCutClass2CoulWolfManager::set_charges(
    std::shared_ptr<std::vector<types::charge>> &charges_)
  {
    born_potential.set_charges(charges_);
    coul_wolf_potential.set_charges(charges_);
    PairBaseManager::set_charges(charges_);
  }



  void
  PairBornCutClass2CoulWolfManager::declare_interactions(
    const types::atom_type     i_atom_type,
    const types::atom_type     j_atom_type,
    const InteractionTypes     interaction,
    const std::vector<double> &parameters)
  {
    if (interaction == InteractionTypes::Class2)
      class2_potential.declare_interactions(i_atom_type,
                                            j_atom_type,
                                            interaction,
                                            parameters);
    else
      {
        born_potential.declare_interactions(i_atom_type,
                                            j_atom_type,
                                            InteractionTypes::Born,
                                            parameters);
        coul_wolf_potential.declare_interactions(i_atom_type,
                                                 j_atom_type,
                                                 InteractionTypes::Coul_Wolf);
      }
  }

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
