
#include <deal.II/base/data_out_base.h>

#ifdef DEAL_II_WITH_ZLIB
#include <zlib.h>
#endif


namespace dealiiqc
{

  using namespace dealii;

  // The following is taken directly from dealii/source/base/data_out_base.cc
  // source file without any alteration.

  namespace
  {
#ifdef DEAL_II_WITH_ZLIB
    // the functions in this namespace are
    // taken from the libb64 project, see
    // http://sourceforge.net/projects/libb64
    //
    // libb64 has been placed in the public
    // domain
    namespace base64
    {
      typedef enum
      {
        step_A, step_B, step_C
      } base64_encodestep;

      typedef struct
      {
        base64_encodestep step;
        char result;
      } base64_encodestate;

      void base64_init_encodestate(base64_encodestate *state_in)
      {
        state_in->step = step_A;
        state_in->result = 0;
      }

      inline
      char base64_encode_value(char value_in)
      {
        static const char *encoding
          = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (value_in > 63) return '=';
        return encoding[(int)value_in];
      }

      int base64_encode_block(const char *plaintext_in,
                              int length_in,
                              char *code_out,
                              base64_encodestate *state_in)
      {
        const char *plainchar = plaintext_in;
        const char *const plaintextend = plaintext_in + length_in;
        char *codechar = code_out;
        char result;

        result = state_in->result;

        switch (state_in->step)
          {
            while (1)
              {
              case step_A:
              {
                if (plainchar == plaintextend)
                  {
                    state_in->result = result;
                    state_in->step = step_A;
                    return codechar - code_out;
                  }
                const char fragment = *plainchar++;
                result = (fragment & 0x0fc) >> 2;
                *codechar++ = base64_encode_value(result);
                result = (fragment & 0x003) << 4;
              }
              case step_B:
              {
                if (plainchar == plaintextend)
                  {
                    state_in->result = result;
                    state_in->step = step_B;
                    return codechar - code_out;
                  }
                const char fragment = *plainchar++;
                result |= (fragment & 0x0f0) >> 4;
                *codechar++ = base64_encode_value(result);
                result = (fragment & 0x00f) << 2;
              }
              case step_C:
              {
                if (plainchar == plaintextend)
                  {
                    state_in->result = result;
                    state_in->step = step_C;
                    return codechar - code_out;
                  }
                const char fragment = *plainchar++;
                result |= (fragment & 0x0c0) >> 6;
                *codechar++ = base64_encode_value(result);
                result  = (fragment & 0x03f) >> 0;
                *codechar++ = base64_encode_value(result);
              }
            }
          }
        /* control should not reach here */
        return codechar - code_out;
      }

      int base64_encode_blockend(char *code_out, base64_encodestate *state_in)
      {
        char *codechar = code_out;

        switch (state_in->step)
          {
          case step_B:
            *codechar++ = base64_encode_value(state_in->result);
            *codechar++ = '=';
            *codechar++ = '=';
            break;
          case step_C:
            *codechar++ = base64_encode_value(state_in->result);
            *codechar++ = '=';
            break;
          case step_A:
            break;
          }
        *codechar++ = '\0';

        return codechar - code_out;
      }
    }


    /**
     * Do a base64 encoding of the given data.
     *
     * The function allocates memory as
     * necessary and returns a pointer to
     * it. The calling function must release
     * this memory again.
     */
    char *
    encode_block (const char *data,
                  const int   data_size)
    {
      base64::base64_encodestate state;
      base64::base64_init_encodestate(&state);

      char *encoded_data = new char[2*data_size+1];

      const int encoded_length_data
        = base64::base64_encode_block (data, data_size,
                                       encoded_data, &state);
      base64::base64_encode_blockend (encoded_data + encoded_length_data,
                                      &state);

      return encoded_data;
    }


    /**
     * Convert between the enum specified inside VtkFlags and the preprocessor
     * constant defined by zlib.
     */
    int get_zlib_compression_level(const DataOutBase::VtkFlags::ZlibCompressionLevel level)
    {
      switch (level)
        {
        case (DataOutBase::VtkFlags::no_compression):
          return Z_NO_COMPRESSION;
        case (DataOutBase::VtkFlags::best_speed):
          return Z_BEST_SPEED;
        case (DataOutBase::VtkFlags::best_compression):
          return Z_BEST_COMPRESSION;
        case (DataOutBase::VtkFlags::default_compression):
          return Z_DEFAULT_COMPRESSION;
        default:
          Assert(false, ExcNotImplemented());
          return Z_NO_COMPRESSION;
        }
    }

    /**
     * Do a zlib compression followed
     * by a base64 encoding of the
     * given data. The result is then
     * written to the given stream.
     */
    template <typename T>
    void write_compressed_block (const std::vector<T>        &data,
                                 const DataOutBase::VtkFlags &flags,
                                 std::ostream                &output_stream)
    {
      if (data.size() != 0)
        {
          // allocate a buffer for compressing
          // data and do so
          uLongf compressed_data_length
            = compressBound (data.size() * sizeof(T));
          char *compressed_data = new char[compressed_data_length];
          int err = compress2 ((Bytef *) compressed_data,
                               &compressed_data_length,
                               (const Bytef *) &data[0],
                               data.size() * sizeof(T),
                               get_zlib_compression_level(flags.compression_level));
          (void)err;
          Assert (err == Z_OK, ExcInternalError());

          // now encode the compression header
          const uint32_t compression_header[4]
            = { 1,                                   /* number of blocks */
                (uint32_t)(data.size() * sizeof(T)), /* size of block */
                (uint32_t)(data.size() * sizeof(T)), /* size of last block */
                (uint32_t)compressed_data_length
              }; /* list of compressed sizes of blocks */

          char *encoded_header = encode_block ((char *)&compression_header[0],
                                               4 * sizeof(compression_header[0]));
          output_stream << encoded_header;
          delete[] encoded_header;

          // next do the compressed
          // data encoding in base64
          char *encoded_data = encode_block (compressed_data,
                                             compressed_data_length);
          delete[] compressed_data;

          output_stream << encoded_data;
          delete[] encoded_data;
        }
    }
#endif
  }
}
