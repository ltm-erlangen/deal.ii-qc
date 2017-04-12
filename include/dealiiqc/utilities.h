
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

namespace dealiiqc
{
  /**
   * Custom types used inside dealiiqc.
   */
  namespace types
  {
    /**
     * The type used for global indices of atoms.
     * In order to have 64-bit unsigned integers (more than 4 billion),
     *  build deal.II with support for 64-bit integers.
     *  The data type always indicates an unsigned integer type.
     */
    typedef  dealii::types::global_dof_index global_atom_index;

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

  } //typedefs

  /**
   * Make sure that sscanf doesn't pickup spaces as unsigned char
   * while parsing atom data stream.
   */
#define UC_SCANF_STR "%hhu"

  namespace Utilities
  {
    using namespace dealii;

    /**
     * Function to check if a Point @p p is a within a certain
     * @p distance from the vertices of a given parent cell.
     */
    template<int dim>
    inline
    bool
    is_point_within_distance_from_cell_vertices( const Point<dim> &p,
                                                 const typename Triangulation<dim>::cell_iterator cell,
                                                 const double &distance)
    {
      // Throw exception if the given cell is is not in a valid
      // cell iterator state.
      AssertThrow( cell->state() == IteratorState::valid,
                   ExcMessage( "The given cell iterator is not in a valid iterator state"));

      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        if (  (cell->vertex(v)- p).norm_square()
              < dealii::Utilities::fixed_power<2>( distance ) )
          return true;
      return false;
    }

    /**
     * Utility function that returns true if a point @p p is outside a bounding box.
     * The box is specified by two points @p minp and @p maxp (the order of
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

    template <int dim, typename Cell>
    inline
    double calculate_cell_radius(const Cell &cell)
    {
      double res = 0.;
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        res = std::max(res, ( (cell->vertex(v)-cell->center()).norm_square() ));
      return std::sqrt(res);
    }


  } // Utilities

}

#endif /* __dealii_qc_utility_h */
