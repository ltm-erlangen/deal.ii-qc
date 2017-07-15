
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>


DEAL_II_QC_NAMESPACE_OPEN


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



  template <int dim, int atomicity, int spacedim>
  void
  WeightsByBase<dim, atomicity, spacedim>::
  initialize (const Triangulation<dim, spacedim> &triangulation,
              const Quadrature<dim>              &quadrature)
  {
    const QTrapez<dim> *q_trapez_ptr =
      dynamic_cast<const QTrapez<dim> *> (&quadrature);

    AssertThrow (q_trapez_ptr != NULL, ExcNotImplemented());

    // TODO: Generalize sampling points by adding more sampling points.
    //       Using quadrature get sampling points from cells.
    //       Adjust the following code to work with
    //       generalized sampling points.
    //       But for now:

    // Add triangulation's vertices as sampling points.
    sampling_points = triangulation.get_vertices();

    // Get locally relevant ghost cells
    /* TODO: This class is not aware of ghost_cell_layer_thickness
    const auto ghost_cells =
      GridTools::
      compute_ghost_cell_layer_within_distance (triangulation,
                                                ghost_cell_layer_thickness);
    */

    // Initialize cells_to_sampling_indices.
    for (types::CellIteratorType<dim, spacedim>
         cell  = triangulation.begin_active();
         cell != triangulation.end();
         cell++)
      {
        // If the cell is not locally relevant, jump.
        /* TODO:
        if (std::find (ghost_cells.begin(),
                       ghost_cells.end(),
                       cell) == ghost_cells.end() && !cell->is_locally_owned())
          continue;
        */

        std::set<unsigned int> this_cell_sampling_indices;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          this_cell_sampling_indices.insert(cell->vertex_index(v));

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto face = cell->face(f);
            if (face->has_children())
              for (unsigned int sf = 0; sf < face->number_of_children(); ++sf)
                {
                  const auto subface = face->child(sf);
                  for (unsigned int
                       v = 0;
                       v < GeometryInfo<dim>::vertices_per_face;
                       v++)
                    this_cell_sampling_indices.insert(subface->vertex_index(v));
                  AssertThrow (!subface->has_children(),
                               ExcInternalError());
                }
          }
        cells_to_sampling_indices[cell] = this_cell_sampling_indices;
      }

  }


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


DEAL_II_QC_NAMESPACE_CLOSE
