
#include <deal.II-qc/atom/data_out_atom_data.h>

#include "data_out_base.cc"

namespace dealiiqc
{
  namespace DataOutBase
  {

    using VtkFlags = dealii::DataOutBase::VtkFlags;

    /**
     * A class to stream in data into vtp files
     */
    class VtpStream
    {
    public:
      VtpStream ( std::ostream &stream,
                  const VtkFlags &flags)
        :
        stream (stream), flags (flags)
      {}

      /**
       * Push point @p p into @see data.
       */
      template <int dim>
      void write_point (const Point<dim> &p);

      void write_scalar(const double);

      /**
       * If libz is found during
       * configuration, this function
       * compresses and encodes the
       * entire points data.
       */
      void flush ();

      /**
       * Forwarding of an output stream.
       */
      template <typename T>
      std::ostream &operator<< (const T &t);

      /**
       * Forwarding of output stream.
       *
       * If libz was found during
       * configuration, this operator
       * compresses and encodes the
       * entire data
       * block. Otherwise, it simply
       * writes it element by
       * element.
       */
      template <typename T>
      std::ostream &operator<< (const std::vector<T> &);

    private:

      /**
       * The ostream to use. Since the life span of these objects is small, we use
       * a very simple storage technique.
       */
      std::ostream &stream;

      /**
       * The flags controlling the output.
       */
      const VtkFlags flags;

      /**
       * A list of points
       * to be used in case we
       * want to compress the data.
       *
       * The data types of these
       * arrays needs to match what
       * we print in the XML-preamble
       * to the respective parts of
       * VTP files (e.g. Float64 and
       * Int32)
       */
      std::vector<double>  data;
    };

    //--------------------------------------------------------------------//

    template<int dim>
    void
    VtpStream::write_point( const Point<dim> &p)
    {
#ifndef DEAL_II_WITH_ZLIB
      // write out coordinates
      stream << p;
      // fill with zeroes
      for (unsigned int i=dim; i<3; ++i)
        stream << " 0";
      stream << '\n';
#else
      // if we want to compress, then
      // first collect all the data in
      // an array
      for (unsigned int i=0; i<dim; ++i)
        data.push_back(p[i]);
      for (unsigned int i=dim; i<3; ++i)
        data.push_back(0);
#endif
    }

    void
    VtpStream::write_scalar( const double d)
    {
#ifndef DEAL_II_WITH_ZLIB
      // write out coordinates
      stream << d << '\n';
#else
      data.push_back(d);
#endif
    }

    void
    VtpStream::flush ()
    {
#ifdef DEAL_II_WITH_ZLIB
      // compress the data we have in
      // memory and write them to the
      // stream. then release the data
      *this << data << '\n';
      data.clear ();
#endif
    }

    template <typename T>
    std::ostream &
    VtpStream::operator<< (const T &t)
    {
      stream << t;
      return stream;
    }


    template <typename T>
    std::ostream &
    VtpStream::operator<< (const std::vector<T> &data)
    {
#ifdef DEAL_II_WITH_ZLIB
      // compress the data we have in
      // memory and write them to the
      // stream. then release the data
      write_compressed_block (data, flags, stream);
#else
      for (unsigned int i=0; i<data.size(); ++i)
        stream << data[i] << ' ';
#endif
      return stream;
    }
  } // namespace DataOutBase


  namespace
  {
    /**
     * Write out into ostream object @p out the vtp header
     */
    void write_vtp_header( std::ostream &out,
                           const DataOutBase::VtkFlags &flags)
    {
      AssertThrow (out, ExcIO());
      out << "<?xml version=\"1.0\" ?> \n";
      out << "<!-- \n";
      out << "# vtk DataFile Version 3.0"
          << '\n';
      if ( flags.print_date_and_time == true)
        out << "#This file was generated by the deal.II-qc library"
            << " on " << dealii::Utilities::System::get_time()
            << " at " << dealii::Utilities::System::get_date()
            << "\n";

      out << "-->\n";
      out << "<VTKFile type=\"PolyData\" version=\"0.1\"";
#ifdef DEAL_II_WITH_ZLIB
      out << " compressor=\"vtkZLibDataCompressor\"";
#endif
#ifdef DEAL_II_WORDS_BIGENDIAN
      out << " byte_order=\"BigEndian\"";
#else
      out << " byte_order=\"LittleEndian\"";
#endif
      out << ">";
      out << '\n';
      out << " <PolyData>";
      out << '\n';
    }

    /**
     * Write out into ostream object @p out the vtp footer.
     */
    void write_vtp_footer( std::ostream &out)
    {
      AssertThrow (out, ExcIO());
      out << " </PolyData>\n";
      out << "</VTKFile>\n";
    }


    template<int dim>
    void write_vtp_main( const typename DataOutAtomData<dim>::CellAtomContainerType &cell_atom_container,
                         const DataOutBase::VtkFlags &flags,
                         std::ostream &out)
    {
      AssertThrow( cell_atom_container.size() > 0,
                   ExcMessage("No atom data to write"));

      DataOutBase::VtpStream vtp_out( out, flags);

#ifdef DEAL_II_WITH_ZLIB
      const char *ascii_or_binary = "binary";
#else
      const char *ascii_or_binary = "ascii";
#endif
      types::global_atom_index n_locally_owned_atoms =0;
      for ( const auto &cell_atom : cell_atom_container)
        if ( cell_atom.first->is_locally_owned() )
          n_locally_owned_atoms++;

      out << "<Piece NumberOfPoints=\"" << n_locally_owned_atoms << "\" >\n";
      out << "  <Points>\n";
      out << "    <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\""
          << ascii_or_binary << "\">\n";

      // Fill Points: the number of components
      // is set to 3.
      // If dim < 3, fill other dimensions with 0s.
      for ( const auto &cell_atom : cell_atom_container)
        if ( cell_atom.first->is_locally_owned() )
          vtp_out.write_point( cell_atom.second.position);


      vtp_out.flush();

      out << "    </DataArray>\n"
          << "  </Points>\n\n";

      out << "  <PointData Scalars=\"Cluster_Weights\">\n"
          << "    <DataArray type=\"Float64\" Name= \"Cluster_Weights\" format=\""
          << ascii_or_binary << "\">\n";

      for ( const auto &cell_atom : cell_atom_container)
        if ( cell_atom.first->is_locally_owned() )
          vtp_out.write_scalar( cell_atom.second.cluster_weight);

      vtp_out.flush();

      out << "    </DataArray>"
          << "  </PointData>"
          << " </Piece>\n";

    }

  } //namespace


  //----------------------------------------------------------------------//
  template<int dim>
  void DataOutAtomData<dim>::write_vtp( const CellAtomContainerType &cell_atom_container,
                                        const dealii::DataOutBase::VtkFlags &flags,
                                        std::ostream &out)
  {
    write_vtp_header( out, flags);
    write_vtp_main<dim>( cell_atom_container, flags, out);
    write_vtp_footer( out);
  }

  template<int dim>
  void DataOutAtomData<dim>::write_pvtp_record( const std::vector<std::string> &vtp_file_names,
                                                const dealii::DataOutBase::VtkFlags &flags,
                                                std::ostream &out)
  {
    AssertThrow (out, ExcIO());
    out << "<?xml version=\"1.0\" ?> \n";
    out << "<!-- \n";
    out << "# vtk DataFile Version 3.0"
        << '\n';

    if ( flags.print_date_and_time == true)
      out << "#This file was generated by the deal.II-qc library"
          << " on " << dealii::Utilities::System::get_time()
          << " at " << dealii::Utilities::System::get_date()
          << "\n";

    out << "-->\n";
    out << "<VTKFile type=\"PPolyData\" version=\"0.1\"";
#ifdef DEAL_II_WITH_ZLIB
    out << " compressor=\"vtkZLibDataCompressor\"";
#endif
#ifdef DEAL_II_WORDS_BIGENDIAN
    out << " byte_order=\"BigEndian\"";
#else
    out << " byte_order=\"LittleEndian\"";
#endif
    out << ">";
    out << '\n';
    out << " <PPolyData>";
    out << '\n';
#ifdef DEAL_II_WITH_ZLIB
    const char *ascii_or_binary = "binary";
#else
    const char *ascii_or_binary = "ascii";
#endif
    out << "  <PPoints>\n";
    out << "    <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\""
        << ascii_or_binary << "\">\n";
    out << "    </PDataArray>\n"
        << "  </PPoints>\n\n";
    out << "  <PPointData Scalars=\"Cluster_Weights\">\n";
    out << "    <PDataArray type=\"Float64\" Name=\"Cluster_Weights\" format=\""
        << ascii_or_binary << "\">\n";
    out << "    </PDataArray>\n"
        << "  </PPointData>\n\n";

    for ( auto name : vtp_file_names)
      out << "  <Piece Source=\"" << name << "\"/>\n";

    out << " </PPolyData>\n";
    out << "</VTKFile>\n";
  }

  template class DataOutAtomData<1>;
  template class DataOutAtomData<2>;
  template class DataOutAtomData<3>;
}



