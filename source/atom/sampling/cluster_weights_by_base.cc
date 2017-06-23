
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>

namespace dealiiqc
{

  namespace Cluster
  {



    template <int dim, int atomicity, int spacedim>
    WeightsByBase<dim, atomicity, spacedim>::
    WeightsByBase (const double &cluster_radius,
                   const double &maximum_cutoff_radius)
      :
      cluster_radius(cluster_radius),
      maximum_cutoff_radius(maximum_cutoff_radius)
    {}



    template <int dim, int atomicity, int spacedim>
    WeightsByBase<dim, atomicity, spacedim>::~WeightsByBase()
    {}



#define SINGLE_WEIGHTS_BY_BASE_INSTANTIATION(DIM, ATOMICITY, SPACEDIM) \
  template class WeightsByBase< DIM, ATOMICITY, SPACEDIM >;            \
   
#define WEIGHTS_BY_BASE(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,         \
              SINGLE_WEIGHTS_BY_BASE_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X              \
   
    // WeightsByBase class Instantiations.
    INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_BASE)

#undef SINGLE_WEIGHTS_BY_BASE_INSTANTIATION
#undef WEIGHTS_BY_BASE


  } // namespace Cluster


} // namespace dealiiqc

