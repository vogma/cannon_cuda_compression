#include "ndzip_api.h"
#include <boost/program_options.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <io/io.hh>
#include <iostream>
#include <memory>

NDZIP_API::NDZIP_API() {

  cudaMalloc(&d_compressed_buffer_size, sizeof(unsigned int));

  size = ndzip::extent(static_cast<ndzip::dim_type>(1));

  size[0] = 139497932;
  req = ndzip::compressor_requirements(size);

  cuda_compressor = ndzip::make_cuda_compressor<double>(req, nullptr);

  cuda_decompressor = ndzip::make_cuda_decompressor<double>(1, nullptr);
}

NDZIP_API::~NDZIP_API() {
  // free(buffer);
  cudaFree(d_compressed_buffer_size);
}

int NDZIP_API::compress_buffer(double *d_buffer, int buffer_size,
                               uint64_t *d_compressed_buffer) {
  size[0] = buffer_size / sizeof(double);
  cuda_compressor->compress(d_buffer, size, d_compressed_buffer,
                            d_compressed_buffer_size);

  int h_compressed_buffer_size;
  // copy 4-Byte uint32_t for compressed buffer length from device to host
  cudaMemcpy(&h_compressed_buffer_size, d_compressed_buffer_size,
             sizeof(unsigned int), cudaMemcpyDeviceToHost);
  return h_compressed_buffer_size;
}

double *NDZIP_API::decompress_buffer(uint64_t *compressed_device_buffer,
                                     double *uncompressed_device_buffer,
                                     size_t buffer_size) {
  size[0] = buffer_size / sizeof(double);
  cuda_decompressor->decompress(compressed_device_buffer,
                                uncompressed_device_buffer, size);

  return uncompressed_device_buffer;
}
