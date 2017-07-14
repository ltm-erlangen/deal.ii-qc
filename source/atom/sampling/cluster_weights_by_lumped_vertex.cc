
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II-qc/atom/cell_molecule_tools.h>

#include <deal.II-qc/atom/sampling/cluster_weights_by_lumped_vertex.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{


  template <int dim, int atomicity, int spacedim>
  WeightsByLumpedVertex<dim, atomicity, spacedim>::
  WeightsByLumpedVertex (const double &cluster_radius,
                         const double &maximum_cutoff_radius)
    :
    WeightsByBase<dim, atomicity, spacedim> (cluster_radius,
                                             maximum_cutoff_radius)
  {}



  template <int dim, int atomicity, int spacedim>
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
  WeightsByLumpedVertex<dim, atomicity, spacedim>::
  update_cluster_weights
  (const dealii::DoFHandler<dim, spacedim>                          &mesh,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const
  {
    // Prepare energy molecules in this container.
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    cell_energy_molecules;

    // Get the squared_energy_radius to identify energy molecules.
    const double squared_energy_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::maximum_cutoff_radius +
       WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Get the squared_cluster_radius to identify cluster molecules.
    const double squared_cluster_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Get underlying p::Triangulation to construct DoFHandler
    const parallel::Triangulation<dim, spacedim> *const ptria =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>
      (&mesh.get_triangulation());

    // Get a consistent MPI_Comm.
    const MPI_Comm &mpi_communicator = ptria != nullptr
                                       ?
                                       ptria->get_communicator()
                                       :
                                       MPI_COMM_SELF;

    // using linear mapping and linear scalar-valued FE
    MappingQ1<dim, spacedim> mapping;
    FE_Q<dim, spacedim> fe(1);

    DoFHandler<dim, spacedim> dof_handler(*ptria);
    dof_handler.distribute_dofs(fe);

    // Get the total number of dofs, in the current case of using linear
    // scalar-valued FE this is exactly equal to the number of vertices.
    const unsigned int n_dofs = dof_handler.n_dofs();

    // Prepare b_I entries of b in this container.
    // Clusters are identified with the global dof indices.
    std::vector<double> b(n_dofs,0);

    // Prepare the A_II diagonal entries of A in this container.
    std::vector<double> A(n_dofs,0);

    // Container to store quadrature points and weights.
    // The quadrature points live in dim-dimensional space.
    std::vector<Point<dim>> points;
    std::vector<double> weights_per_molecule;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    // Gather global indices of the local dofs here for a given cell.
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    for (types::DoFCellIteratorType<dim, spacedim>
         cell  = dof_handler.begin_active();
         cell != dof_handler.end();
         cell++)
      {
        // Include all the molecules associated to this active cell as
        // quadrature points. The quadrature points will be then used to
        // initialize fe_values object so as to evaluate the shape function
        // values at the all the lattice sites in the atomistic system.

        // Get cell molecules range
        const auto cell_molecules_range =
          CellMoleculeTools::
          molecules_range_in_cell<dim, atomicity, spacedim> (cell,
                                                             cell_molecules);

        const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
        &cell_molecules_range_begin = cell_molecules_range.first.first,
         &cell_molecules_range_end  = cell_molecules_range.first.second;

        // Prepare the total number of molecules in this cell here.
        // This is also the total number of quadrature points in this cell.
        const unsigned int n_molecules_in_current_cell =
          cell_molecules_range.second;

        // If this cell is not within the locally relevant active cells of the
        // current MPI process continue active cell loop
        if (n_molecules_in_current_cell == 0)
          continue;

        // Resize containers to known number of energy molecules in cell.
        points.resize(n_molecules_in_current_cell);
        weights_per_molecule.resize(n_molecules_in_current_cell);

        types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
        cell_molecule_iterator = cell_molecules_range_begin;
        for (unsigned int
             q = 0;
             q < n_molecules_in_current_cell;
             q++, cell_molecule_iterator++)
          {
            Molecule<spacedim, atomicity> molecule =
              cell_molecule_iterator->second;

            // Update dim-dimensional quadrature point using
            // pseudo spacedim-dimensional
            // molecule.position_inside_reference_cell.
            for (int d = 0; d < dim; ++d)
              points[q][d] = molecule.position_inside_reference_cell[d];

            // Check the proximity of the molecule to it's associated
            // cell's vertices.
            const auto closest_vertex =
              Utilities::find_closest_vertex (molecule_initial_location(molecule),
                                              cell);

            if (closest_vertex.second < squared_energy_radius)
              {
                if (closest_vertex.second < squared_cluster_radius)
                  {
                    // molecule is cluster molecule
                    molecule.cluster_weight = 1.;
                    weights_per_molecule[q] = 1.;
                  }
                else
                  {
                    // molecule is not cluster molecule
                    molecule.cluster_weight = 0.;
                    weights_per_molecule[q] = 0.;
                  }

                // Insert molecule into cell_energy_molecules if it is within a distance of
                // energy_radius to associated cell's vertices.
                cell_energy_molecules.insert(std::make_pair(cell, molecule));
              }

          }

        Assert (cell_molecule_iterator == cell_molecules_range_end,
                ExcMessage("The number of energy molecules in the cell counted "
                           "using the distance between the iterator ranges "
                           "yields a different result than "
                           "incrementing the iterator to cell_energy_molecules."
                           "Why wasn't this error thrown earlier?"));

        Assert (points.size() == weights_per_molecule.size(),
                ExcDimensionMismatch(points.size(), weights_per_molecule.size()));

        // Do not need to compute b_I and A_I for ghost cells as we will
        // later sum contributions from all the processes.
        if (!cell->is_locally_owned())
          continue;

        // Now we are ready to initialize FEValues object for this cell.
        // Unfortunately, we need to set up a new FEValues object
        // as location of molecules associated to the cell is generally different
        // for each cell.
        FEValues<dim, spacedim>
        fe_values (mapping,
                   fe,
                   Quadrature<dim> (points, weights_per_molecule),
                   update_values);
        fe_values.reinit (cell);

        cell->get_dof_indices(local_dofs);

        for (unsigned int i=0; i<local_dofs.size(); ++i)
          {
            const dealii::types::global_dof_index I = local_dofs[i];
            for (unsigned int q = 0; q < n_molecules_in_current_cell; q++)
              {
                b[I] += fe_values.shape_value(i,q);
                A[I] += fe_values.shape_value(i,q)*
                        fe_values.get_quadrature().weight(q);
              }
          }

      } // end of the loop over all active cells

    //---Finished adding energy molecules

    // Accumulate b entries per vertex from all MPI processes.
    dealii::Utilities::MPI::sum (b, mpi_communicator, b);

    // Accumulate A diagonal entries per vertex from all MPI processes.
    dealii::Utilities::MPI::sum (A, mpi_communicator, A);

    //---Finished updating b_per_cell and A_per_cell

    //---Now update cluster weights with correct value

    // Loop over all cells, and loop over energy molecules within each cell
    // update their weights by multiplying with the factor
    // (b_per_cell/A_per_cell)
    for (types::DoFCellIteratorType<dim, spacedim>
         dof_cell  = dof_handler.begin_active();
         dof_cell != dof_handler.end();
         dof_cell++)
      {
        const types::CellIteratorType<dim, spacedim> cell = dof_cell;

        auto cell_energy_molecule_range =
          cell_energy_molecules.equal_range(cell);

        // Either the cell does not have any molecules associated to it or
        // the cell is not locally relevant to the current MPI process.
        if (cell_energy_molecule_range.first==cell_energy_molecule_range.second)
          continue;

        // Loop over all energy molecules within this cell
        for (auto
             cell_energy_molecule_iter  = cell_energy_molecule_range.first;
             cell_energy_molecule_iter != cell_energy_molecule_range.second;
             cell_energy_molecule_iter++)
          {
            Molecule<spacedim, atomicity>  &molecule =
              cell_energy_molecule_iter->second;

            // Get the closest vertex (of this cell) to the molecule.
            const auto vertex_and_squared_distance =
              Utilities::find_closest_vertex (molecule_initial_location(molecule),
                                              cell);

            // We need to get the global dof index from the local index of
            // the closest vertex.

            // we use scalar Q1 FEM, so we have one DoF per vertex,
            // thus as the second parameter to vertex_dof_index()
            // we provide zero.
            const dealii::types::global_dof_index I =
              dof_cell->vertex_dof_index(vertex_and_squared_distance.first,0);

            Assert (A[I] != 0, ExcInternalError());

            //TODO: For each cell, this factor could be cached
            //      for all its vertices.
            // The cluster weight was previously set to 1. if the molecule is
            // cluster molecule and 0. if the molecule is not cluster molecule.
            molecule.cluster_weight *= b[I]
                                       /
                                       A[I];
          }
      }

    return cell_energy_molecules;
  }



#define SINGLE_WEIGHTS_BY_LUMPED_VERTEX_INSTANTIATION(DIM, ATOMICITY, SPACEDIM)\
  template class WeightsByLumpedVertex< DIM, ATOMICITY, SPACEDIM >;            \
   
#define WEIGHTS_BY_LUMPED_VERTEX(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,                  \
              SINGLE_WEIGHTS_BY_LUMPED_VERTEX_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X                       \
   
  // WeightsByLumpedVertex class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_LUMPED_VERTEX)

#undef SINGLE_WEIGHTS_BY_LUMPED_VERTEX_INSTANTIATION
#undef WEIGHTS_BY_LUMPED_VERTEX


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
