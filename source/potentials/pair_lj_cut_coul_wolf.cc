
#include <deal.II-qc/potentials/pair_lj_cut_coul_wolf.h>

#include <cmath>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{
  PairLJCutCoulWolfManager::PairLJCutCoulWolfManager(
    const double &alpha,
    const double &coul_cutoff_radius,
    const double &lj_cutoff_radius,
    const bool    with_tail,
    const double &factor_coul)
    : lj_potential(lj_cutoff_radius, with_tail)
    , coul_wolf_potential(alpha, coul_cutoff_radius, factor_coul)
  {}



  void
  PairLJCutCoulWolfManager::set_charges(
    std::shared_ptr<std::vector<types::charge>> &charges_)
  {
    lj_potential.set_charges(charges_);
    coul_wolf_potential.set_charges(charges_);
    PairBaseManager::set_charges(charges_);
  }


  void
  PairLJCutCoulWolfManager::declare_interactions(
    const types::atom_type     i_atom_type,
    const types::atom_type     j_atom_type,
    InteractionTypes           interaction,
    const std::vector<double> &parameters)
  {
    Assert(interaction == InteractionTypes::LJ_Coul_Wolf,
           ExcMessage("Invalid InteractionTypes specified"));

    lj_potential.declare_interactions(i_atom_type,
                                      j_atom_type,
                                      InteractionTypes::LJ,
                                      parameters);

    coul_wolf_potential.declare_interactions(i_atom_type,
                                             j_atom_type,
                                             InteractionTypes::Coul_Wolf);
    DEAL_II_QC_UNUSED_VARIABLE(interaction);
  }

} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
