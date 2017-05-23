
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>

namespace dealiiqc
{

  namespace Cluster
  {



    template <int dim>
    WeightsByBase<dim>::WeightsByBase (const double &cluster_radius,
                                       const double &maximum_cutoff_radius)
      :
      cluster_radius(cluster_radius),
      maximum_cutoff_radius(maximum_cutoff_radius)
    {}



    template <int dim>
    WeightsByBase<dim>::~WeightsByBase()
    {}



    // Instantiations
    template class WeightsByBase<1>;
    template class WeightsByBase<2>;
    template class WeightsByBase<3>;



  } // namespace Cluster


} // namespace dealiiqc

