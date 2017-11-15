
#ifndef __dealii_qc_tests_h
#define __dealii_qc_tests_h

#include <algorithm>
#include <limits>
#include <type_traits>
#include <fstream>

#include <deal.II/base/mpi.h>
#include <deal.II/base/logstream.h>


namespace Testing
{

  // Return true if the two floating point numbers x and y are  close to each
  // other.
  template<typename T>
  typename std::enable_if<std::is_floating_point<T>::value, bool>::type
  almost_equal (const T &x,
                const T &y,
                const unsigned int ulp)
  {
    // The machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= (std::numeric_limits<T>::epsilon()
                              * std::max(std::fabs(x), std::fabs(y))
                              * ulp)
           ||
           std::fabs(x-y) < std::numeric_limits<T>::min();
  }


  // A direct copy from deal.II test.h file
  // for pseudo-random int generation.
  int rand(const bool reseed=false,
           const int seed=1)
  {
    static int r[32];
    static int k;
    static bool inited=false;
    if (!inited || reseed)
      {
        //srand treats a seed 0 as 1 for some reason
        r[0]=(seed==0)?1:seed;

        for (int i=1; i<31; i++)
          {
            r[i] = (16807LL * r[i-1]) % 2147483647;
            if (r[i] < 0)
              r[i] += 2147483647;
          }
        k=31;
        for (int i=31; i<34; i++)
          {
            r[k%32] = r[(k+32-31)%32];
            k=(k+1)%32;
          }

        for (int i=34; i<344; i++)
          {
            r[k%32] = r[(k+32-31)%32] + r[(k+32-3)%32];
            k=(k+1)%32;
          }
        inited=true;
        if (reseed==true)
          return 0;// do not generate new no
      }

    r[k%32] = r[(k+32-31)%32] + r[(k+32-3)%32];
    int ret = r[k%32];
    k=(k+1)%32;
    return (unsigned int)ret >> 1;
  }



  /**
   * A class to write output of MPI processes into separate files and
   * then write the content of the files into a single ostream object.
   */
  class SequentialFileStream
  {
  public:

    /**
     * Constructor takes an MPI_Communicator object.
     */
    SequentialFileStream (const MPI_Comm &mpi_communicator)
      :
      mpi_communicator(mpi_communicator),
      this_process(dealii::Utilities::MPI::this_mpi_process (mpi_communicator)),
      n_processes(dealii::Utilities::MPI::n_mpi_processes (mpi_communicator))
    {
      std::string deallogname = "output" + dealii::Utilities::int_to_string(this_process);
      logfile.open (deallogname);
      dealii::deallog.attach(logfile, /*do not print job id*/ false);
      dealii::deallog.depth_console(0);
    }



    /**
     * Destructor.
     */
    ~SequentialFileStream ()
    {
      logfile.close();
      MPI_Barrier(mpi_communicator);

      if (this_process==0)
        for (unsigned int p=0; p<n_processes; ++p)
          {
            std::string deallogname = "output" + dealii::Utilities::int_to_string(p);
            std::ifstream f(deallogname);
            std::string line;
            while (std::getline(f, line))
              std::cout << p << ":" << line << std::endl;
          }
    }

  private:

    /**
     * MPI_Comm object that determines which processes are involved
     * in (current) communication.
     */
    const MPI_Comm mpi_communicator;

    /**
     * Current process.
     */
    const unsigned int this_process;

    /**
     * Total number of processes in #mpi_communicator.
     */
    const unsigned int n_processes;

    /**
     * File stream object to store output of each process.
     */
    std::ofstream logfile;

  };



  /**
   * Go through the input stream @p in and filter out binary data for the key @p key .
   * The filtered stream is returned in @p out.
   *
   * Copied shamelessly from dealii/tests/tests.h
   */
  void filter_out_xml_key(std::istream &in, const std::string &key, std::ostream &out)
  {
    std::string line;
    bool found = false;
    const std::string opening = "<" + key;
    const std::string closing = "</" + key;
    while (std::getline(in, line))
      {
        if (line.find(opening) != std::string::npos &&
            line.find("binary") != std::string::npos)
          {
            found = true;
            // remove everything after ">" but keep things after "</"
            const auto pos = line.find(closing);
            if (pos != std::string::npos)
              {
                line = line.substr(0, line.find(">", 0)+1) + line.substr(pos);
                found = false;
              }
            else
              line = line.substr(0, line.find(">", 0)+1);
            out << line << std::endl;
          }
        else if (line.find(closing) != std::string::npos)
          {
            found = false;
            // remove everything before "<"
            line = line.substr(line.find("<",0));
            out << line << std::endl;
          }
        else if (!found)
          out << line << std::endl;
      }
  }

} // namespace Testing



#endif // __dealii_qc_tests_h
