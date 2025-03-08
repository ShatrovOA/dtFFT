module dtfft_interface_nvcomp
!! dtFFT Interfaces to nvCOMP library
use iso_c_binding
use dtfft_parameters, only: dtfft_stream_t
implicit none
private

public :: nvcompBatchedCascadedCompressAsync, nvcompBatchedCascadedDecompressAsync
public :: nvcompBatchedLZ4CompressAsync, nvcompBatchedLZ4DecompressAsync
public :: nvcompBatchedSnappyCompressAsync, nvcompBatchedSnappyDecompressAsync
public :: nvcompBatchedZstdCompressAsync, nvcompBatchedZstdDecompressAsync
public :: nvcompBatchedCascadedCompressGetMaxOutputChunkSize, nvcompBatchedCascadedCompressGetTempSize
public :: nvcompBatchedLZ4CompressGetMaxOutputChunkSize, nvcompBatchedLZ4CompressGetTempSize
public :: nvcompBatchedSnappyCompressGetMaxOutputChunkSize, nvcompBatchedSnappyCompressGetTempSize
public :: nvcompBatchedZstdCompressGetMaxOutputChunkSize, nvcompBatchedZstdCompressGetTempSize

  !! Structure for Cascaded compression options
  type, bind(C) :: nvcompBatchedCascadedOpts_t
    integer(c_int) :: num_RLEs         !! Number of RLE passes
    integer(c_int) :: num_deltas       !! Number of delta encodings
    integer(c_int) :: use_bp           !! Use bit-packing (0 or 1)
  end type nvcompBatchedCascadedOpts_t

  !! Structure for LZ4 compression options (placeholder, unused in batched API)
  type, bind(C) :: nvcompBatchedLZ4Opts_t
    integer(c_int) :: unused           !! Unused in batched API
  end type nvcompBatchedLZ4Opts_t

  !! Structure for Snappy compression options (placeholder, unused in batched API)
  type, bind(C) :: nvcompBatchedSnappyOpts_t
    integer(c_int) :: unused           !! Unused in batched API
  end type nvcompBatchedSnappyOpts_t

  !! Structure for Zstd compression options (placeholder, unused in batched API)
  type, bind(C) :: nvcompBatchedZstdOpts_t
    integer(c_int) :: unused           !! Unused in batched API
  end type nvcompBatchedZstdOpts_t

  interface
    function nvcompBatchedCascadedCompressGetMaxOutputChunkSize(chunk_size, opts, max_compressed_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedCascadedCompressGetMaxOutputChunkSize")
    !! Computes the maximum possible size of a compressed chunk for the Cascaded compression algorithm.
    import
      integer(c_size_t), value :: chunk_size          !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts                !! Pointer to Cascaded compression options
      integer(c_size_t)        :: max_compressed_size !! Returned maximum compressed chunk size
      integer(c_int)           :: nvcompStatus        !! NVCOMP API result code (0 = success)
    end function nvcompBatchedCascadedCompressGetMaxOutputChunkSize

    function nvcompBatchedCascadedCompressGetTempSize(num_chunks, chunk_size, opts, temp_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedCascadedCompressGetTempSize")
    !! Calculates the size of the temporary buffer needed for batched Cascaded compression.
    import
      integer(c_size_t), value :: num_chunks  !! Number of chunks to compress
      integer(c_size_t), value :: chunk_size  !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts        !! Pointer to Cascaded compression options
      integer(c_size_t)        :: temp_size   !! Returned size of the temporary buffer
      integer(c_int)           :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedCascadedCompressGetTempSize

    function nvcompBatchedCascadedCompressAsync(in_ptrs, in_sizes, max_chunk_size, num_chunks, &
                                                temp_ptr, temp_size, out_ptrs, out_sizes, opts, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedCascadedCompressAsync")
    !! Compresses multiple chunks of data asynchronously using the Cascaded algorithm on the GPU.
    import
      type(c_ptr), value       :: in_ptrs         !! Array of pointers to input data (GPU)
      type(c_ptr), value       :: in_sizes        !! Array of input chunk sizes
      integer(c_size_t), value :: max_chunk_size  !! Maximum size of an uncompressed chunk
      integer(c_size_t), value :: num_chunks      !! Number of chunks to compress
      type(c_ptr), value       :: temp_ptr        !! Pointer to temporary buffer (GPU)
      integer(c_size_t), value :: temp_size       !! Size of the temporary buffer
      type(c_ptr), value       :: out_ptrs        !! Array of pointers to output compressed data (GPU)
      type(c_ptr), value       :: out_sizes       !! Array of output sizes (in: expected, out: actual)
      type(c_ptr), value       :: opts            !! Pointer to Cascaded compression options
      type(dtfft_stream_t)     :: stream          !! Stream identifier for asynchronous execution
      integer(c_int)           :: nvcompStatus    !! NVCOMP API result code (0 = success)
    end function nvcompBatchedCascadedCompressAsync

    function nvcompBatchedCascadedDecompressAsync(comp_ptrs, comp_sizes, out_sizes, num_chunks, &
                                                  temp_ptr, temp_size, out_ptrs, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedCascadedDecompressAsync")
    !! Decompresses multiple chunks of data asynchronously using the Cascaded algorithm on the GPU.
    import
      type(c_ptr),          value :: comp_ptrs    !! Array of pointers to compressed data (GPU)
      type(c_ptr),          value :: comp_sizes   !! Array of compressed chunk sizes
      type(c_ptr),          value :: out_sizes    !! Array of uncompressed chunk sizes
      integer(c_size_t),    value :: num_chunks   !! Number of chunks to decompress
      type(c_ptr),          value :: temp_ptr     !! Pointer to temporary buffer (GPU)
      integer(c_size_t),    value :: temp_size    !! Size of the temporary buffer
      type(c_ptr),          value :: out_ptrs     !! Array of pointers to output decompressed data (GPU)
      type(dtfft_stream_t), value :: stream       !! Stream identifier for asynchronous execution
      integer(c_int)              :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedCascadedDecompressAsync

    function nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, opts, max_compressed_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedLZ4CompressGetMaxOutputChunkSize")
    !! Computes the maximum possible size of a compressed chunk for the LZ4 compression algorithm.
    import
      integer(c_size_t),    value :: chunk_size          !! Maximum size of an uncompressed chunk
      type(c_ptr),          value :: opts                !! Pointer to LZ4 compression options (unused)
      integer(c_size_t)           :: max_compressed_size !! Returned maximum compressed chunk size
      integer(c_int)              :: nvcompStatus        !! NVCOMP API result code (0 = success)
    end function nvcompBatchedLZ4CompressGetMaxOutputChunkSize

    function nvcompBatchedLZ4CompressGetTempSize(num_chunks, chunk_size, opts, temp_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedLZ4CompressGetTempSize")
    !! Calculates the size of the temporary buffer needed for batched LZ4 compression.
    import
      integer(c_size_t),    value :: num_chunks  !! Number of chunks to compress
      integer(c_size_t),    value :: chunk_size  !! Maximum size of an uncompressed chunk
      type(c_ptr),          value :: opts        !! Pointer to LZ4 compression options (unused)
      integer(c_size_t)           :: temp_size   !! Returned size of the temporary buffer
      integer(c_int)              :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedLZ4CompressGetTempSize

    function nvcompBatchedLZ4CompressAsync(in_ptrs, in_sizes, max_chunk_size, num_chunks, &
                                           temp_ptr, temp_size, out_ptrs, out_sizes, opts, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedLZ4CompressAsync")
    !! Compresses multiple chunks of data asynchronously using the LZ4 algorithm on the GPU.
    import
      type(c_ptr),          value :: in_ptrs         !! Array of pointers to input data (GPU)
      type(c_ptr),          value :: in_sizes        !! Array of input chunk sizes
      integer(c_size_t),    value :: max_chunk_size  !! Maximum size of an uncompressed chunk
      integer(c_size_t),    value :: num_chunks      !! Number of chunks to compress
      type(c_ptr),          value :: temp_ptr        !! Pointer to temporary buffer (GPU)
      integer(c_size_t),    value :: temp_size       !! Size of the temporary buffer
      type(c_ptr),          value :: out_ptrs        !! Array of pointers to output compressed data (GPU)
      type(c_ptr),          value :: out_sizes       !! Array of output sizes (in: expected, out: actual)
      type(c_ptr),          value :: opts            !! Pointer to LZ4 compression options (unused)
      type(dtfft_stream_t), value :: stream          !! Stream identifier for asynchronous execution
      integer(c_int)              :: nvcompStatus    !! NVCOMP API result code (0 = success)
    end function nvcompBatchedLZ4CompressAsync

    function nvcompBatchedLZ4DecompressAsync(comp_ptrs, comp_sizes, out_sizes, num_chunks, &
                                             temp_ptr, temp_size, out_ptrs, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedLZ4DecompressAsync")
    !! Decompresses multiple chunks of data asynchronously using the LZ4 algorithm on the GPU.
    import
      type(c_ptr),          value :: comp_ptrs    !! Array of pointers to compressed data (GPU)
      type(c_ptr),          value :: comp_sizes   !! Array of compressed chunk sizes
      type(c_ptr),          value :: out_sizes    !! Array of uncompressed chunk sizes
      integer(c_size_t),    value :: num_chunks   !! Number of chunks to decompress
      type(c_ptr),          value :: temp_ptr     !! Pointer to temporary buffer (GPU)
      integer(c_size_t),    value :: temp_size    !! Size of the temporary buffer
      type(c_ptr),          value :: out_ptrs     !! Array of pointers to output decompressed data (GPU)
      type(dtfft_stream_t), value :: stream       !! Stream identifier for asynchronous execution
      integer(c_int)              :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedLZ4DecompressAsync

    function nvcompBatchedSnappyCompressGetMaxOutputChunkSize(chunk_size, opts, max_compressed_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedSnappyCompressGetMaxOutputChunkSize")
    !! Computes the maximum possible size of a compressed chunk for the Snappy compression algorithm.
    import
      integer(c_size_t), value :: chunk_size          !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts                !! Pointer to Snappy compression options (unused)
      integer(c_size_t)        :: max_compressed_size !! Returned maximum compressed chunk size
      integer(c_int)           :: nvcompStatus        !! NVCOMP API result code (0 = success)
    end function nvcompBatchedSnappyCompressGetMaxOutputChunkSize

    function nvcompBatchedSnappyCompressGetTempSize(num_chunks, chunk_size, opts, temp_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedSnappyCompressGetTempSize")
    !! Calculates the size of the temporary buffer needed for batched Snappy compression.
    import
      integer(c_size_t), value :: num_chunks  !! Number of chunks to compress
      integer(c_size_t), value :: chunk_size  !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts        !! Pointer to Snappy compression options (unused)
      integer(c_size_t)        :: temp_size   !! Returned size of the temporary buffer
      integer(c_int)           :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedSnappyCompressGetTempSize

    function nvcompBatchedSnappyCompressAsync(in_ptrs, in_sizes, max_chunk_size, num_chunks, &
                                              temp_ptr, temp_size, out_ptrs, out_sizes, opts, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedSnappyCompressAsync")
    !! Compresses multiple chunks of data asynchronously using the Snappy algorithm on the GPU.
    import
      type(c_ptr), value       :: in_ptrs         !! Array of pointers to input data (GPU)
      type(c_ptr), value       :: in_sizes        !! Array of input chunk sizes
      integer(c_size_t), value :: max_chunk_size  !! Maximum size of an uncompressed chunk
      integer(c_size_t), value :: num_chunks      !! Number of chunks to compress
      type(c_ptr), value       :: temp_ptr        !! Pointer to temporary buffer (GPU)
      integer(c_size_t), value :: temp_size       !! Size of the temporary buffer
      type(c_ptr), value       :: out_ptrs        !! Array of pointers to output compressed data (GPU)
      type(c_ptr), value       :: out_sizes       !! Array of output sizes (in: expected, out: actual)
      type(c_ptr), value       :: opts            !! Pointer to Snappy compression options (unused)
      type(dtfft_stream_t)     :: stream          !! Stream identifier for asynchronous execution
      integer(c_int)           :: nvcompStatus    !! NVCOMP API result code (0 = success)
    end function nvcompBatchedSnappyCompressAsync

    function nvcompBatchedSnappyDecompressAsync(comp_ptrs, comp_sizes, out_sizes, num_chunks, &
                                                temp_ptr, temp_size, out_ptrs, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedSnappyDecompressAsync")
    !! Decompresses multiple chunks of data asynchronously using the Snappy algorithm on the GPU.
    import
      type(c_ptr), value       :: comp_ptrs    !! Array of pointers to compressed data (GPU)
      type(c_ptr), value       :: comp_sizes   !! Array of compressed chunk sizes
      type(c_ptr), value       :: out_sizes    !! Array of uncompressed chunk sizes
      integer(c_size_t), value :: num_chunks   !! Number of chunks to decompress
      type(c_ptr), value       :: temp_ptr     !! Pointer to temporary buffer (GPU)
      integer(c_size_t), value :: temp_size    !! Size of the temporary buffer
      type(c_ptr), value       :: out_ptrs     !! Array of pointers to output decompressed data (GPU)
      type(dtfft_stream_t)     :: stream       !! Stream identifier for asynchronous execution
      integer(c_int)           :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedSnappyDecompressAsync

    function nvcompBatchedZstdCompressGetMaxOutputChunkSize(chunk_size, opts, max_compressed_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedZstdCompressGetMaxOutputChunkSize")
    !! Computes the maximum possible size of a compressed chunk for the Zstd compression algorithm.
    import
      integer(c_size_t), value :: chunk_size          !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts                !! Pointer to Zstd compression options (unused)
      integer(c_size_t)        :: max_compressed_size !! Returned maximum compressed chunk size
      integer(c_int)           :: nvcompStatus        !! NVCOMP API result code (0 = success)
    end function nvcompBatchedZstdCompressGetMaxOutputChunkSize

    function nvcompBatchedZstdCompressGetTempSize(num_chunks, chunk_size, opts, temp_size) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedZstdCompressGetTempSize")
    !! Calculates the size of the temporary buffer needed for batched Zstd compression.
    import
      integer(c_size_t), value :: num_chunks  !! Number of chunks to compress
      integer(c_size_t), value :: chunk_size  !! Maximum size of an uncompressed chunk
      type(c_ptr), value       :: opts        !! Pointer to Zstd compression options (unused)
      integer(c_size_t)        :: temp_size   !! Returned size of the temporary buffer
      integer(c_int)           :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedZstdCompressGetTempSize

    function nvcompBatchedZstdCompressAsync(in_ptrs, in_sizes, max_chunk_size, num_chunks, &
                                            temp_ptr, temp_size, out_ptrs, out_sizes, opts, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedZstdCompressAsync")
    !! Compresses multiple chunks of data asynchronously using the Zstd algorithm on the GPU.
    import
      type(c_ptr), value       :: in_ptrs         !! Array of pointers to input data (GPU)
      type(c_ptr), value       :: in_sizes        !! Array of input chunk sizes
      integer(c_size_t), value :: max_chunk_size  !! Maximum size of an uncompressed chunk
      integer(c_size_t), value :: num_chunks      !! Number of chunks to compress
      type(c_ptr), value       :: temp_ptr        !! Pointer to temporary buffer (GPU)
      integer(c_size_t), value :: temp_size       !! Size of the temporary buffer
      type(c_ptr), value       :: out_ptrs        !! Array of pointers to output compressed data (GPU)
      type(c_ptr), value       :: out_sizes       !! Array of output sizes (in: expected, out: actual)
      type(c_ptr), value       :: opts            !! Pointer to Zstd compression options (unused)
      type(dtfft_stream_t)     :: stream          !! Stream identifier for asynchronous execution
      integer(c_int)           :: nvcompStatus    !! NVCOMP API result code (0 = success)
    end function nvcompBatchedZstdCompressAsync

    function nvcompBatchedZstdDecompressAsync(comp_ptrs, comp_sizes, out_sizes, num_chunks, &
                                              temp_ptr, temp_size, out_ptrs, stream) &
      result(nvcompStatus) &
      bind(C, name="nvcompBatchedZstdDecompressAsync")
    !! Decompresses multiple chunks of data asynchronously using the Zstd algorithm on the GPU.
    import
      type(c_ptr), value       :: comp_ptrs    !! Array of pointers to compressed data (GPU)
      type(c_ptr), value       :: comp_sizes   !! Array of compressed chunk sizes
      type(c_ptr), value       :: out_sizes    !! Array of uncompressed chunk sizes
      integer(c_size_t), value :: num_chunks   !! Number of chunks to decompress
      type(c_ptr), value       :: temp_ptr     !! Pointer to temporary buffer (GPU)
      integer(c_size_t), value :: temp_size    !! Size of the temporary buffer
      type(c_ptr), value       :: out_ptrs     !! Array of pointers to output decompressed data (GPU)
      type(dtfft_stream_t)     :: stream       !! Stream identifier for asynchronous execution
      integer(c_int)           :: nvcompStatus !! NVCOMP API result code (0 = success)
    end function nvcompBatchedZstdDecompressAsync

  end interface

end module dtfft_interface_nvcomp