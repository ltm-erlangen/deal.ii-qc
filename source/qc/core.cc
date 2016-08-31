// a source file which contains definition of core functions of QC class
#include <dealiiqc/qc.h>
#include <deal.II/grid/grid_tools.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  QC<dim>::~QC ()
  {
    dof_handler.clear();
  }

  template <int dim>
  QC<dim>::QC (/*const Parameters<dim> &parameters*/)
    :
    mpi_communicator(MPI_COMM_WORLD),
    pcout (std::cout,
           (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    triangulation (mpi_communicator,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    fe    (FE_Q<dim>(1),dim),
    dof_handler    (triangulation),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::never,
                     TimerOutput::wall_times)
  {
    // TODO: read from input file
    const unsigned int N = 4;
    atoms.resize(N+1);
    const double L = 1.;
    for (unsigned int i = 0; i <= N; i++)
      {
        Point<dim> p;
        p[0] = (L*i)/N;
        atoms[i].position = p;
      }
  }

  template <int dim>
  void QC<dim>::setup_system ()
  {
    TimerOutput::Scope t (computing_timer, "Setup system");

    dof_handler.distribute_dofs (fe);

    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_set);

    // set-up constraints objects
    constraints.reinit (locally_relevant_set);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);

    /*
    std::set<types::boundary_id>       dirichlet_boundary_ids;
    typename FunctionMap<dim>::type    dirichlet_boundary_functions;
    ZeroFunction<dim>                  homogeneous_dirichlet_bc (1);
    dirichlet_boundary_ids.insert(0);
    dirichlet_boundary_functions[0] = &homogeneous_dirichlet_bc;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    */
    constraints.close ();
  }


  template <int dim>
  void QC<dim>::run ()
  {
    pcout << "Quasic-continuum simulations in " << dim <<"D." << std::endl;

    // TODO: read .gmsh file
    {
      GridGenerator::hyper_cube (triangulation);
      triangulation.refine_global(1);
    }

    setup_system();
    associate_atoms_with_cells();

    computing_timer.print_summary();
  }


  template <int dim>
  void QC<dim>::associate_atoms_with_cells ()
  {
    TimerOutput::Scope t (computing_timer, "Associate atoms with cells");

    for (auto a = atoms.begin(); a != atoms.end(); ++a)
      {
        const std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>
        my_pair = GridTools::find_active_cell_around_point(mapping, dof_handler, a->position);

        a->reference_position = GeometryInfo<dim>::project_to_unit_cell(my_pair.second);
        a->parent_cell = my_pair.first;
      }
  }

  // instantiations:
  template void QC<1>::run ();
  template void QC<2>::run ();
  template void QC<3>::run ();
  template void QC<1>::setup_system ();
  template void QC<2>::setup_system ();
  template void QC<3>::setup_system ();
  template QC<1>::~QC ();
  template QC<2>::~QC ();
  template QC<3>::~QC ();
  template QC<1>::QC ();
  template QC<2>::QC ();
  template QC<3>::QC ();
  template void QC<1>::associate_atoms_with_cells ();
  template void QC<2>::associate_atoms_with_cells ();
  template void QC<3>::associate_atoms_with_cells ();
}
