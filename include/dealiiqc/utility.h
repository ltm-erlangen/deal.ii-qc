
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>
#include <cmath>
#include <functional>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>

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

    using namespace dealii;
    /**
     * Utility function that returns true if a point @p p is outside a bounding box.
     * The box is specified by two points @p minp and @p maxp (the order of
     * specifying points is important).
     */
    template<int dim>
    bool
    is_outside_bounding_box( const Point<dim> &minp,
                             const Point<dim> &maxp,
                             const Point<dim> &p)
    {
      bool outside = false;
      for (unsigned int d=0; d<dim; ++d)
        if ( (minp[d] > p[d]) || (p[d] > maxp[d]) )
          {
            outside = true;
            break;
          }

      return outside;
    }

    /**
     * Returns cell radius, the distance to the farthest
     * vertex from the center of the cell, of a given @p cell.
     */
    template < class MeshType>
    inline
    double calculate_cell_radius(const typename MeshType::active_cell_iterator &cell)
    {
      double res = 0.;
      for (unsigned int v=0; v<dealii::GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
        res = std::max(res, ( cell->vertex(v)).distance(cell->center()) );
      return res;
    }

  } // Utilities

}

#endif /* __dealii_qc_utility_h */
