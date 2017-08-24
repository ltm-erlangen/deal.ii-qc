
#include <deal.II-qc/grid/shared_tria.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>


DEAL_II_QC_NAMESPACE_OPEN


#ifdef DEAL_II_WITH_MPI
namespace parallel
{
  namespace shared
  {

    template <int dim, int spacedim>
    Triangulation<dim,spacedim>::Triangulation (MPI_Comm mpi_communicator,
                                                const typename dealii::Triangulation<dim,spacedim>::MeshSmoothing smooth_grid,
                                                const bool allow_artificial_cells,
                                                const dealii::parallel::shared::Triangulation::Settings settings,
                                                const double ghost_cell_layer_thickness)
      :
      dealii::parallel::shared::Triangulation<dim,spacedim> (mpi_communicator,
                                                             smooth_grid,
                                                             allow_artificial_cells,
                                                             settings),
      ghost_cell_layer_thickness(ghost_cell_layer_thickness)
    {
      Assert (ghost_cell_layer_thickness >= 0.,
              ExcMessage("Invalid ghost cells layer thickness specified."));
    }



    template <int dim, int spacedim>
    void Triangulation<dim,spacedim>::setup_ghost_cells()
    {
      if (ghost_cell_layer_thickness > 0. && this->allow_artificial_cells)
        {
          // --- Gather ghost cells within specified thickness.

          const std::vector<types::CellIteratorType<dim, spacedim> >
          active_halo_layer_vector =
            dealii::GridTools::compute_ghost_cell_layer_within_distance
            (*static_cast<dealii::parallel::shared::Triangulation<dim, spacedim>*>(this),
             ghost_cell_layer_thickness);

          std::set<types::CellIteratorType<dim,spacedim> >
          ghost_cells(active_halo_layer_vector.begin(), active_halo_layer_vector.end());

          types::CellIteratorType<dim, spacedim>
          cell = this->begin_active(),
          endc = this->end();

          for (unsigned int index=0; cell != endc; cell++, index++)
            {
              // store original/true subdomain ids:
              this->true_subdomain_ids_of_cells[index] = cell->subdomain_id();

              if (cell->is_locally_owned() == false &&
                  ghost_cells.find(cell) == ghost_cells.end())
                cell->set_subdomain_id(dealii::numbers::artificial_subdomain_id);
            }
        }

      this->update_number_cache();
    }


#define SINGLE_SHARED_TRIA_INSTANTIATION(DIM, SPACEDIM) \
  template class Triangulation< DIM, SPACEDIM >;        \
   
#define SHARED_TRIA(R, X)                       \
  BOOST_PP_IF(IS_DIM_AND_SPACEDIM_PAIR_VALID X, \
              SINGLE_SHARED_TRIA_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(2)) X          \
   
    // Triangulation class Instantiations.
    INSTANTIATE_WITH_DIM_AND_SPACEDIM(SHARED_TRIA)

#undef SINGLE_SHARED_TRIA_INSTANTIATION
#undef SHARED_TRIA

  } // shared
} // parallel

#endif
DEAL_II_QC_NAMESPACE_CLOSE
