
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <fstream>

# include <boost/preprocessor/facilities/empty.hpp>
# include <boost/preprocessor/list/at.hpp>
# include <boost/preprocessor/list/for_each_product.hpp>
# include <boost/preprocessor/tuple/elem.hpp>
# include <boost/preprocessor/tuple/to_list.hpp>

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/numbers.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>



// Preprocessor definitions for instantiations.
#define       _DIM_   BOOST_PP_TUPLE_TO_LIST(3,  (1,2,3))
#define  _SPACE_DIM_  BOOST_PP_TUPLE_TO_LIST(3,  (1,2,3))
#define  _ATOMICITY_  BOOST_PP_TUPLE_TO_LIST(10, (1,2,3,4,5,6,7,8,9,10))



// Accessors for _SPACE_DIM_, _ATOMICITY_ in (two element tuple) X.
#define  FIRST_OF_TWO_IS_SPACEDIM(X)    BOOST_PP_TUPLE_ELEM(2, 0, X)
#define SECOND_OF_TWO_IS_ATOMICITY(X)   BOOST_PP_TUPLE_ELEM(2, 1, X)



/**
 * A macro function that returns 1 if @p _DIM_ is less or equal to @p _SPACE_DIM_
 * and 0 otherwise among three entries _DIM_, _ATOMICITY_ and, _SPACE_DIM_.
 */
#define IS_DIM_LESS_EQUAL_SPACEDIM(_DIM_, _ATOMICITY_, _SPACE_DIM_) \
  BOOST_PP_LESS_EQUAL(_DIM_, _SPACE_DIM_)



/**
 * A macro function that returns 1 if @p _DIM_ is less or equal to @p _SPACE_DIM_
 * and 0 otherwise.
 */
#define IS_DIM_AND_SPACEDIM_PAIR_VALID(_DIM_, _SPACE_DIM_) \
  BOOST_PP_LESS_EQUAL(_DIM_, _SPACE_DIM_)



/**
 * A macro to expand CLASS macro using (_DIM_, _SPACE_DIM_) tuples.
 */
#define INSTANTIATE_WITH_DIM_AND_SPACEDIM(CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 2, (_DIM_, _SPACE_DIM_))



/**
 * A macro to expand CLASS macro using (_SPACE_DIM_, _ATOMICITY_) tuples.
 */
#define INSTANTIATE_WITH_SPACEDIM_AND_ATOMICITY(R, CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 2, (_SPACE_DIM_, _ATOMICITY_))



/**
 * A macro to expand CLASS macro using (_DIM_, _SPACE_DIM_, _ATOMICITY_) tuples.
 */
#define INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(CLASS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(CLASS, 3, (_DIM_, _ATOMICITY_, _SPACE_DIM_))



/**
 * Macro to suppress unused variable warnings.
 */
#define DEAL_II_QC_UNUSED_VARIABLE(X) ((void)(X))



/***************************************************************************
 * Two macro names that we put at the top and bottom of all deal.II-qc files
 * and that will be expanded to "namespace dealiiqc {" and "}".
 */
#define DEAL_II_QC_NAMESPACE_OPEN   namespace dealiiqc {
#define DEAL_II_QC_NAMESPACE_CLOSE  }


DEAL_II_QC_NAMESPACE_OPEN


/**
 * Custom types used inside dealiiqc.
 */
namespace types
{
  /**
   * The type used for global indices of atoms.
   * In order to have 64-bit unsigned integers (more than 4 billion),
   * build deal.II with support for 64-bit integers.
   * The data type always indicates an unsigned integer type.
   */
  typedef  dealii::types::global_dof_index global_atom_index;

  /**
   * The type used for global indices of molecules.
   */
  typedef  global_atom_index global_molecule_index;

  // TODO: Use of correct charge units; Use charge_t for book keeping.
  /**
   * The type used for storing charge of the atom.
   * Computations with charge of atoms don't need high precision.
   */
  typedef float charge;

  /**
   * The type used for identifying atom types. The enumeration starts
   * from 0.
   */
  typedef unsigned char atom_type;

  /**
   * A typedef for Triangulation's active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using CellIteratorType =
    typename dealii::Triangulation<dim, spacedim>::active_cell_iterator;

  /**
   * A typedef for DoFHandler's active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using DoFCellIteratorType =
    typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;

  /**
   * A typedef for DoFHandler's const active_cell_iterator for ease of use.
   */
  template<int dim, int spacedim=dim>
  using ConstCellIteratorType =
    const CellIteratorType<dim, spacedim>;

} //typedefs



/**
 * A namespace for certain fixed numbers.
 */
namespace numbers
{
  /**
   * A number representing invalid cluster weight.
   */
  static const double
  invalid_cluster_weight = std::numeric_limits<double>::signaling_NaN();

} // numbers



/**
 * Make sure that sscanf doesn't pickup spaces as unsigned char
 * while parsing atom data stream.
 */
#define UC_SCANF_STR "%hhu"

namespace Utilities
{
  using namespace dealii;



  // TODO: Add a test for this function.
  /**
   * Find the closest vertex of a given cell @p cell to a given Point @p and
   * return a pair of its number and the squared distance.
   */
  template<int dim>
  inline
  std::pair<unsigned int, double>
  find_closest_vertex (const Point<dim> &p,
                       const typename Triangulation<dim>::cell_iterator cell)
  {
    // Throw exception if the given cell is is not in a valid
    // cell iterator state.
    AssertThrow (cell->state() == IteratorState::valid,
                 ExcMessage ("The given cell iterator is not in a "
                             "valid iterator state"));

    // Assume the first vertex is the closest at first.
    double squared_distance = p.distance_square(cell->vertex(0));
    unsigned int vertex_number = 0;

    // Loop over all the other vertices to search for closest vertex.
    for (unsigned int v=1; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const double p_squared_distance = p.distance_square(cell->vertex(v));
        if (p_squared_distance < squared_distance)
          {
            squared_distance = p_squared_distance;
            vertex_number = v;
          }
      }
    return std::make_pair(vertex_number, squared_distance);
  }



  /**
   * Find the closest point in a given list of @p points to a point @p p
   * and return a pair of its (the closest point's) index in the list and
   * the squared distance of separation.
   */
  template<int dim>
  inline
  std::pair<unsigned int, double>
  find_closest_point (const Point<dim>              &p,
                      const std::vector<Point<dim>> &points)
  {
    AssertThrow (points.size(),
                 ExcMessage("The given list of points is empty."));

    // Assume the first point is the closest at first.
    double squared_distance = points[0].distance_square(p);
    unsigned int point_index = 0;

    // Loop over all the other points to search for closest points.
    for (unsigned int i=1; i<points.size(); ++i)
      {
        const double p_squared_distance = points[i].distance_square(p);
        if (p_squared_distance < squared_distance)
          {
            squared_distance = p_squared_distance;
            point_index = i;
          }
      }

    return std::make_pair(point_index, squared_distance);
  }



  /**
   * Utility function that returns true if a point @p p is outside a bounding
   * box. The box is specified by two points @p minp and @p maxp (the order of
   * specifying points is important).
   */
  template<int dim>
  inline
  bool
  is_outside_bounding_box( const Point<dim> &minp,
                           const Point<dim> &maxp,
                           const Point<dim> &p)
  {
    for (unsigned int d=0; d<dim; ++d)
      if ( (minp[d] > p[d]) || (p[d] > maxp[d]) )
        {
          return true;
        }

    return false;
  }



  /**
   * Given a string that contains a list of @p minor_delimiter separated
   * text separated by a @p major_delimiter, split it into its components,
   * remove leading and trailing spaces.
   *
   * The input string can end without specifying any delimiters (possibly
   * followed by an arbitrary amount of whitespace). For example,
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l; ", ';', ',');
   * @endcode
   * yields the same 3-element list of sub-lists output
   * <code>{ {"abc", "def"},{"ghi"}, {"j", "k", "l"}}</code>
   * as you would get if the input had been
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l", ';', ',');
   * @endcode
   * or
   * @code
   *   Utilities::split_list_of_string_lists
   *   ("abc, def; ghi; j, k, l;", ';', ',');
   * @endcode
   */
  inline
  std::vector<std::vector<std::string> >
  split_list_of_string_lists (const std::string &s,
                              const char major_delimiter = ';',
                              const char minor_delimiter = ',')
  {
    AssertThrow (major_delimiter!=minor_delimiter,
                 ExcMessage("Invalid major and minor delimiters provided!"));

    std::vector<std::vector<std::string>> res;

    const std::vector<std::string> coeffs_per_type =
      dealii::Utilities::split_string_list (s,
                                            major_delimiter);
    res.resize(coeffs_per_type.size());
    for (unsigned int i = 0; i < coeffs_per_type.size(); ++i)
      res[i] = dealii::Utilities::split_string_list (coeffs_per_type[i],
                                                     minor_delimiter);
    return res;
  }



  /**
   * Return the range of atom types on parsing a given string
   * @p numeric_string that consists a possible wildcard asterisk and the
   * total number of atom types @p n_atom_types. There are only five
   * possible ways to describe a range of atom types using @p numeric_string:
   * a) "i"   to describe the range [i, i+1)
   * b) "*"   to describe the range [0, @p max_atom_types)
   * c) "i*"  to describe the range [i, @p max_atom_types)
   * d) "*i"  to describe the range [0, i+1)
   * e) "i*j" to describe the range [i, j+1)
   */
  std::pair<types::atom_type, types::atom_type>
  atom_type_range (const std::string      &numeric_string,
                   const types::atom_type  n_atom_types);



  /**
   * Return the dim-dimensional volume of the hyper-ball segment, the region
   * of the hyper-ball being cutoff by a hyper-plane that is @p d distance
   * from the center of the hyper-ball of radius @p radius.
   *
   * @note The permissible range of @p d is [0, radius].
   * When the value of @p d is 0 the volume of the half hyper-ball
   * is returned.
   */
  template <int dim>
  inline
  double hyperball_segment_volume (const double &radius, const double &d)
  {
    double volume;

    // Height of the segment.
    const double height = radius-d;

    AssertThrow (0 <= height && height <= 2.*radius,
                 ExcMessage("This function is called with invalid parameter: "
                            "d, the distance from the center"
                            "of the hyper-ball to the hyper-plane."
                            "Allowed range of d is [0, radius]."));
    if (dim==1)
      volume = height;
    else if (dim==2)
      {
        // Half of the angle inscribed at the center of the hyper-ball.
        const double alpha = std::acos(d/radius);
        volume = radius* (radius*alpha - d*std::sin(alpha));
      }
    else if (dim==3)
      volume = dealii::numbers::PI * height * height * (3*radius - height)/3.;

    return volume;
  }


  namespace
  {
    inline
    std::string data_out_filename (const std::string                 &name,
                                   const unsigned int                 timestep_no,
                                   const dealii::types::subdomain_id  id,
                                   const std::string                 &suffix)
    {
      return name +
             dealii::Utilities::int_to_string(timestep_no,4) + "." +
             dealii::Utilities::int_to_string(id,3) +
             suffix;
    }
  }

  /**
   * Write out @p solution_vector at @p time time occurring at time step
   * @p time_step into vtu and visit file formats, with filenames
   * prefixed @p solution_base_name, for visualization given a @p dof_handler.
   * The @p data_component_interpretation argument contains information about
   * how the individual components of output files (or @p solution_vector) that
   * consist of more than one data set are to be interpreted.
   */
  template <int dim, typename VectorType, int atomicity=1, int spacedim=dim>
  void write_vector_out (const VectorType                &solution_vector,
                         const DoFHandler<dim, spacedim> &dof_handler,
                         const std::string               &solution_base_name,
                         const double                     time,
                         const unsigned int               timestep_no,
                         const std::vector<DataComponentInterpretation::DataComponentInterpretation > &data_component_interpretation=std::vector<DataComponentInterpretation::DataComponentInterpretation>())
  {
    const Triangulation<dim, spacedim> &triangulation = dof_handler.get_triangulation ();
    const parallel::Triangulation<dim, spacedim> *const ptria =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>
      (&triangulation);

    // Get a consistent MPI_Comm.
    const MPI_Comm &mpi_communicator = ptria != nullptr
                                       ?
                                       ptria->get_communicator()
                                       :
                                       MPI_COMM_SELF;

    const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_mpi_processes  = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    std::vector<std::string> solution_names;

    {
      for (int atom_stamp=0; atom_stamp<atomicity; ++atom_stamp)
        {
          const std::string name = solution_base_name +
                                   dealii::Utilities::int_to_string(atom_stamp, 2);
          for (int d=0; d<dim; ++d)
            solution_names.push_back(name);
        }
    }

    DataOut<dim> data_out;
    data_out.add_data_vector (dof_handler,
                              solution_vector,
                              solution_names,
                              data_component_interpretation);

    std::vector<dealii::types::subdomain_id> partition_int (triangulation.n_active_cells());
    GridTools::get_subdomain_association (triangulation, partition_int);

    const Vector<float> partitioning (partition_int.begin(),
                                      partition_int.end());
    data_out.add_data_vector (partitioning, "partitioning");
    data_out.build_patches ();

    AssertThrow (n_mpi_processes < 1000,
                 ExcNotImplemented());

    const std::string solution_filename  = data_out_filename (solution_base_name,
                                                              timestep_no,
                                                              this_mpi_process,
                                                              ".vtu");

    std::ofstream solution_output (solution_filename.c_str());
    data_out.write_vtu (solution_output);

    if (this_mpi_process==0)
      {
        std::vector<std::string> solution_filenames, atom_data_filenames;
        for (unsigned int i=0; i<n_mpi_processes; ++i)
          solution_filenames.push_back (data_out_filename (solution_base_name,
                                                           timestep_no,
                                                           i,
                                                           ".vtu"));

        const std::string
        visit_master_filename = (solution_base_name +
                                 dealii::Utilities::int_to_string(timestep_no,4) +
                                 ".visit");
        std::ofstream visit_master (visit_master_filename.c_str());
        DataOutBase::write_visit_record (visit_master, solution_filenames);

        const std::string
        pvtu_solution_master_filename = (solution_base_name +
                                         dealii::Utilities::int_to_string(timestep_no,4) +
                                         ".pvtu");
        std::ofstream pvtu_solution_master  (pvtu_solution_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_solution_master,
                                    solution_filenames);

        static std::vector<std::pair<double, std::string> >
        times_and_solution_names;

        times_and_solution_names.push_back (std::make_pair(time,
                                                           pvtu_solution_master_filename.c_str()));
        std::ofstream pvd_solution_output  (solution_base_name  + ".pvd");
        DataOutBase::write_pvd_record (pvd_solution_output,
                                       times_and_solution_names);
      }
  }

} // Utilities


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_utility_h */
