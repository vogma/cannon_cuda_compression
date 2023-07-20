#ifndef NDZIP_API_H
#define NDZIP_API_H

#include <stdint.h>
#include <stdlib.h>
#include <ndzip/cuda.hh>
class NDZIP_API {
public:
  NDZIP_API();
  ~NDZIP_API();
  int compress_buffer(double *d_buffer, int buffer_size,
                      uint64_t *d_compressed_buffer);
  double *decompress_buffer(uint64_t *compressed_device_buffer,
                            double *uncompressed_device_buffer,
                            size_t buffer_size);

private:
  double *buffer;
  unsigned int *d_compressed_buffer_size; // in bytes
  ndzip::extent size;
  ndzip::compressor_requirements req;
  std::unique_ptr<ndzip::cuda_compressor<double>> cuda_compressor;
  std::unique_ptr<ndzip::cuda_decompressor<double>> cuda_decompressor;
};

#endif
