
#ifndef __dealii_qc_cluster_weights_by_vertex_h_
#define __dealii_qc_cluster_weights_by_vertex_h_

#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>

namespace dealiiqc
{

  namespace Cluster
  {

    /**
     * A class which creates a cluster around each vertex and calculates cluster
     * weights using the ratio of atoms within each cluster (vertex) to the
     * number of atoms that are closest to each vertex. The latter is nothing
     * else but the number of atoms in Voronoi cell associated with a given
     * vertex.
     */
    template <int dim>
    class WeightsByVertex : public WeightsByBase <dim>
    {
    public:

      /**
       * Constructor
       */
      WeightsByVertex (const double &cluster_radius,
                       const double &maximum_energy_radius);

      /**
       * @see WeightsByBase::update_cluster_weights()
       * and WeightsByVertex class description.
       */
      types::CellAtomContainerType<dim>
      update_cluster_weights (const types::MeshType<dim> &mesh,
                              const types::CellAtomContainerType<dim> &atoms) const;

    };

  } // namespace Cluster

} // namespace dealiiqc




#endif /* __dealii_qc_cluster_weights_by_vertex_h_ */
