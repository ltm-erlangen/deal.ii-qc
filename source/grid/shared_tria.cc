
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/grid/shared_tria.h>


DEAL_II_QC_NAMESPACE_OPEN


#ifdef DEAL_II_WITH_MPI
namespace parallel
{
  namespace shared
  {
    template <int dim, int spacedim>
    Triangulation<dim, spacedim>::Triangulation(
      MPI_Comm mpi_communicator,
      const typename dealii::Triangulation<dim, spacedim>::MeshSmoothing
                   smooth_grid,
      const double ghost_cell_layer_thickness)
      : dealii::parallel::shared::Triangulation<dim, spacedim>(
          mpi_communicator,
          smooth_grid,
          ghost_cell_layer_thickness >= 0.)
      , ghost_cell_layer_thickness(ghost_cell_layer_thickness)
    {}



    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::setup_ghost_cells()
    {
      if (ghost_cell_layer_thickness > 0.)
        {
          // --- Gather ghost cells within specified thickness.
          dealii::IteratorFilters::LocallyOwnedCell
            locally_owned_cell_predicate;
          std::function<bool(const types::CellIteratorType<dim, spacedim> &)>
            predicate(locally_owned_cell_predicate);

          const std::vector<types::CellIteratorType<dim, spacedim>>
            ghost_cells =
              dealii::GridTools::compute_active_cell_layer_within_distance(
                *static_cast<
                  dealii::parallel::shared::Triangulation<dim, spacedim> *>(
                  this),
                predicate,
                ghost_cell_layer_thickness);

          for (auto cell = this->begin_active(); cell != this->end(); cell++)
            if (cell->is_locally_owned() == false &&
                std::find(ghost_cells.begin(), ghost_cells.end(), cell) ==
                  ghost_cells.end())
              cell->set_subdomain_id(dealii::numbers::artificial_subdomain_id);

          this->update_number_cache();
        }
    }


#  define SINGLE_SHARED_TRIA_INSTANTIATION(_DIM, _SPACE_DIM) \
    template class Triangulation<_DIM, _SPACE_DIM>;

#  define SHARED_TRIA(R, X)                       \
    BOOST_PP_IF(IS_DIM_AND_SPACEDIM_PAIR_VALID X, \
                SINGLE_SHARED_TRIA_INSTANTIATION, \
                BOOST_PP_TUPLE_EAT(2))            \
    X

    // Triangulation class Instantiations.
    INSTANTIATE_WITH_DIM_AND_SPACEDIM(SHARED_TRIA)

#  undef SINGLE_SHARED_TRIA_INSTANTIATION
#  undef SHARED_TRIA

  } // namespace shared
} // namespace parallel

#endif
DEAL_II_QC_NAMESPACE_CLOSE
