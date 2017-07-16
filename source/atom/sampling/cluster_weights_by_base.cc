
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
    maximum_cutoff_radius(maximum_cutoff_radius),
    tria_ptr(NULL)
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
    AssertThrow (dynamic_cast<const QTrapez<dim> *> (&quadrature) != NULL,
                 ExcNotImplemented());

    // TODO: Generalize sampling points by adding more sampling points.
    //       Using quadrature get sampling points from cells.
    //       Adjust the following code to work with
    //       generalized sampling points.
    //       But for now:

    // Initialize tria_ptr data member.
    tria_ptr = &triangulation;

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

        // First, store vertex indices of all vertices of a given cell.
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          this_cell_sampling_indices.insert(cell->vertex_index(v));

        // We also need to pick up all the hanging nodes, if any, on this cell
        // as the sampling points of this cell.
        // That's because if this cell is coarser than one (or more) of its
        // neighbors, we must gather all the molecules inside the cluster
        // sphere around a sampling point associated to the hanging node of the
        // neighboring cell.

        // So we need to pick up all the hanging nodes on this cell by
        // the following logic:
        // cell -> all faces -> all its sub-faces -> all its vertices.
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
                  Assert (!subface->has_children(),
                          ExcInternalError());
                }
          }

        cells_to_sampling_indices[cell] = this_cell_sampling_indices;
      }

    locally_relevant_sampling_indices.clear();

    // FIXME: This will change if, Quadrature<> other than QTrapez<> is used.
    locally_relevant_sampling_indices.set_size(triangulation.n_vertices());

    // Initialize locally relevant sampling indices.
    for (const auto &entry : cells_to_sampling_indices)
      {
        const std::set<unsigned int> &sampling_indices = entry.second;

        // The set of sampling indices stores unique elements and
        // is already sorted, they can be directly added to
        // the locally relevant sampling indices.
        locally_relevant_sampling_indices.add_indices(sampling_indices.begin(),
                                                      sampling_indices.end());
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
