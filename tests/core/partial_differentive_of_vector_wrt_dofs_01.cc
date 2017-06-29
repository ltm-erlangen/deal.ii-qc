#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/generic_linear_algebra.h>

using namespace dealii;



// For a given pair of points a and b in two seperate cells, a test case to
// evaluate $ r^{ab}_{,k} = \gz n^{ab} * [ \gz N_k(X^a) - \gz N_k(X^b)] $ where
// $ \gz N_k $ is a vector valued shape function associated to the global dof k.
// Note that the RHS is a dot product to yield a scalar $ r^{ab}_{,k} $.
// Subsequently, this will generate 12 double values to be compared with blessed
// output in 2D.

// The blessed output is generated using scalar valued shape functions.
// The related maxima script provided at the end.



template<int dim>
using CellPointType =
    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>;

template<int dim>
class Test
{
public:

  Test()
    :
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    fe (FE_Q<dim>(1), dim),
    dof_handler(triangulation)
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back(2);
    for (int i = 1; i < dim; ++i)
      repetitions.push_back(1);

    dealii::Point<dim> p1, p2;

    for (int d = 0; d < dim; ++d)
      p2[d] = 1.;

    p2[0] = 2.;

    //   ___________________
    //   |        |        |
    //   |        |        |
    //   |________|________|
    //
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               p1,
                                               p2,
                                               true);
    std::ofstream f("tria.vtk");
    GridOut().write_vtk (triangulation, f);

    dof_handler.distribute_dofs (fe);

    IndexSet locally_relevant_set;
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_set);
    gradient.reinit(dof_handler.n_dofs());
    // set-up constraints objects
    constraints.reinit (locally_relevant_set);
    constraints.close ();
  }

  void check(const Point<dim> &p1, const Point<dim> &p2)
  {

    const CellPointType<dim> my_pair_1 =
      GridTools::find_active_cell_around_point (mapping,
                                               dof_handler,
                                               p1);

    const CellPointType<dim> my_pair_2 =
      GridTools::find_active_cell_around_point (mapping,
                                                dof_handler,
                                                p2);

    run (my_pair_1, my_pair_2, p1, p2);

    std::cout << "Point 1: "
              << my_pair_1.second
              << "    "
              << "Point 2: "
              << my_pair_2.second
              << " in their reference cells."
              << std::endl;

    for (unsigned int i = 0; i < dof_handler.n_dofs(); i+=dim)
      {
        for (int d = 0; d < dim; ++d)
          std::cout << gradient[i+d] <<  "\t";
        std::cout << std::endl;
      }
  }

  void run (const CellPointType<dim> &my_pair_1,
            const CellPointType<dim> &my_pair_2,
            const Point<dim> &p1,
            const Point<dim> &p2)
  {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    dealii::Vector<double> local_gradient_I, local_gradient_II;
    local_gradient_I.reinit(dofs_per_cell);
    local_gradient_II.reinit(dofs_per_cell);

    local_gradient_I = 0.;
    local_gradient_II = 0.;
    gradient = 0.;

    std::vector<dealii::types::global_dof_index>
    local_dof_indices(dofs_per_cell);

    {
      FEValues<dim>
      fe_value_a (mapping,
                  fe,
                  Quadrature<dim>(std::vector<Point<dim>> (1, {my_pair_1.second})),
                  update_values),
      fe_value_b (mapping,
                  fe,
                  Quadrature<dim>(std::vector<Point<dim>> (1, {my_pair_2.second})),
                  update_values);

      fe_value_a.reinit(my_pair_1.first);
      fe_value_b.reinit(my_pair_2.first);

      Tensor<1,dim> n = (p1-p2) / p1.distance(p2);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          const unsigned int comp = fe.system_to_component_index(k).first;
          local_gradient_I[k]  =  n[comp]*fe_value_a.shape_value(k, 0);
          local_gradient_II[k] = -n[comp]*fe_value_b.shape_value(k, 0);
        }

      my_pair_1.first->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_gradient_I,
                                              local_dof_indices,
                                              gradient);

      my_pair_2.first->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (local_gradient_II,
                                              local_dof_indices,
                                              gradient);
    }
  }

  void write_dofs_and_support_points_info()
  {
    constraints.print(std::cout);

    std::map<types::global_dof_index, Point<dim> > support_points;

    DoFTools::map_dofs_to_support_points (mapping,
                                          dof_handler,
                                          support_points);

    const std::string filename =
      "grid" + Utilities::int_to_string(dim) + ".gp";
    std::ofstream f(filename.c_str());

    f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
      << "set output \"grid" << Utilities::int_to_string(dim) << ".png\"" << std::endl
      << "set size square" << std::endl
      << "set view equal xy" << std::endl
      << "unset xtics" << std::endl
      << "unset ytics" << std::endl
      << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
    GridOut().write_gnuplot (triangulation, f);
    f << "e" << std::endl;

    DoFTools::write_gnuplot_dof_support_point_info(f,
                                                   support_points);
    f << "e" << std::endl;
  }

private:
  parallel::shared::Triangulation<dim> triangulation;
  const MappingQ1<dim>   mapping;
  FESystem<dim>          fe;
  DoFHandler<dim>        dof_handler;
  dealii::Vector<double> gradient;
  ConstraintMatrix       constraints;
};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      Test<2> problem;
      //problem.write_dofs_and_support_points_info();
      {
        Point<2> p1(0.23, 0.37), p2(1.73, 0.43);
        problem.check(p1,p2);
        problem.check(p2,p1);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}

/*

Maxima script:

a : [0.23, 0.37];
b : [1.73, 0.43];

load ("eigen");
n(x) := unitvector(x);
nab : n(a-b);

N(x,y) := [if x < 1 then (1-x)*(1-y) else 0          ,
           if x < 1 then (  x)*(1-y) else (2-x)*(1-y),
           if x < 1 then (1-x)*(  y) else 0          ,
           if x < 1 then (  x)*(  y) else (2-x)*y    ,
           if x < 1 then           0 else (x-1)*(1-y),
           if x < 1 then           0 else (x-1)*(  y) ];

Nk_a : N(a[1], a[2]);
Nk_b : N(b[1], b[2]);

M : zeromatrix (6, 2);

for k:1 thru 6 do
  (
    M[k][1] : nab[1] * (Nk_a[k] - Nk_b[k]),
    M[k][2] : nab[2] * (Nk_a[k] - Nk_b[k])
  );

float(M)

//------------------------------------------------------------------------------
N0(x,y) := (if x < 1 then (1-x)*(1-y) else 0           );
N1(x,y) := (if x < 1 then (  x)*(1-y) else (2-x)*(1-y) );
N2(x,y) := (if x < 1 then (1-x)*(  y) else 0           );
N3(x,y) := (if x < 1 then (  x)*(  y) else (2-x)*y     );
N4(x,y) := (if x < 1 then           0 else (x-1)*(1-y) );
N5(x,y) := (if x < 1 then           0 else (x-1)*(  y) );


Shape functions can be visualized in wxMaxima using the following.

plot3d (N0(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
plot3d (N1(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
plot3d (N2(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
plot3d (N3(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
plot3d (N4(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
plot3d (N5(x,y), [x, 0, 2], [y, 0, 1], [grid, 10, 10], [mesh_lines_color,false])
*/
