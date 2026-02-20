project: dtFFT
src_dir: ../src
    ../src/interfaces/api
    ../src/interfaces/cuda
    ../src/interfaces/fft/cufft
    ../src/interfaces/fft/fftw
    ../src/interfaces/fft/mkl
    ../src/interfaces/fft/vkfft
include: ../src/include
    ../build
    ../include
output_dir: html/
project_github: https://github.com/ShatrovOA/dtFFT
author: Oleg Shatrov
github: https://github.com/ShatrovOA
email: shatrov.oleg.a@gmail.com
summary: DataTyped Fast Fourier Transform
media_dir: ./images
favicon: ./images/favicon.png
md_extensions: markdown.extensions.toc
    markdown.extensions.smarty
    markdown.extensions.extra
    markdown_checklist.extension
docmark: !
macro: DTFFT_WITH_PROFILER
    DTFFT_WITH_CUDA
    DTFFT_WITH_NCCL
    DTFFT_WITH_NVSHMEM
    DTFFT_WITH_CUFFT
    DTFFT_WITH_VKFFT
    DTFFT_WITH_FFTW
    DTFFT_WITH_MKL
    DTFFT_WITH_COMPRESSION
    NCCL_HAVE_COMMREGISTER
    NCCL_HAVE_MEMALLOC
    ENABLE_PERSISTENT_COLLECTIVES
    ENABLE_PERSISTENT_COMM
preprocess: true
preprocessor: gfortran -E
display: public
    protected
    private
warn: false
graph: true
coloured_edges: true
sort: permission
search: false

<img src="./media/pencils.png" alt="pencils" width="850"/>

