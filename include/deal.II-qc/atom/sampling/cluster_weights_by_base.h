
#ifndef __dealii_qc_cluster_weights_by_base_h_
#define __dealii_qc_cluster_weights_by_base_h_

#include <deal.II-qc/atom/atom_data.h>

namespace dealiiqc
{

  namespace Cluster
  {

    /**
     * Base class for assigning @see cluster_weight to atoms
     */
    template <int dim>
    class WeightsByBase
    {
    public:

      /**
       * Constructor.
       */
      WeightsByBase (const double &cluster_radius,
                     const double &maximum_cutoff_radius);


      virtual ~WeightsByBase();

      /**
       * Return energy atoms (in a cell based data structure) with appropriately
       * set cluster weights based on @p atoms that were associated to @p mesh.
       * The cluster radius and the maximum cutoff radius are specified in
       * constructor.
       *
       * The returned energy atoms may have non-zero (cluster atoms) or
       * zero (non cluster atoms) cluster weights.
       *
       * An atom is indeed an energy atom if it is within #cluster_radius
       * plus #maximum_cutoff_radius distance to it's surrounding
       * (or associated) cell's vertices.
       */
      virtual
      types::CellAtomContainerType<dim>
      update_cluster_weights (const types::MeshType<dim> &mesh,
                              const types::CellAtomContainerType<dim> &atoms) const = 0;

    protected:

      /**
       * The cluster radius for the QC approach.
       */
      const double cluster_radius;

      /**
       * The maximum of cutoff radii.
       */
      const double maximum_cutoff_radius;
    };


  } // namespace Cluster


} // namespace dealiiqc



#endif /* __dealii_qc_cluster_weights_by_base_h_ */
