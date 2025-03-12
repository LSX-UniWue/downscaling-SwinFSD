# Use the specified PyTorch base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Run system package updates and install dependencies
RUN apt-get -y update && \
    apt-get -y install \
        libgeos++ \
        libproj-dev \
        libnetcdf-dev \
        libhdf5-dev \
        netcdf-bin \
        libx11-6 \
        libxext6
    

# Install Python dependencies
RUN pip install --no-cache-dir \
    jsonargparse[signatures] \
    lightning \
    matplotlib \
    xarray \
    dask \
    plotly \
    geopandas \
    cartopy \
    netCDF4 \
    h5py \
    wandb \
    basicsr \
    einops \
    "numpy<2.0" \
    diffusers \
    timm