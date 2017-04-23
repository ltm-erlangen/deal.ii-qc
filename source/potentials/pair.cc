
#include <dealiiqc/potentials/pair.h>

namespace dealiiqc
{

  namespace Potential
  {
    template<int dim>
    Pair<dim>::Pair( const ConfigureQC &configure)
    :
    configure_qc(configure)
    {
    }

    template<int dim>
    void
    Pair<dim>::initialize_energy_per_cluster_atom( unsigned int n)
    {
      energy_per_cluster_atom.resize(n, 0.);
    }

    template class Pair<1>;
    template class Pair<2>;
    template class Pair<3>;

  } // namespace Potential

} /* namespace dealiiqc */
