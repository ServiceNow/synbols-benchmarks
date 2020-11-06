FROM nvidia/cuda:10.1-devel-ubuntu18.04
#MAINTAINER TODO

ARG GOOGLE_FONTS_COMMIT=ed61614fb47affd2a4ef286e0b313c5c47226c69

# Install Python 3
RUN apt-get update && \
    apt-get install -y curl python3.8-dev python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    ln -sf $(which python3.8) $(which python3) && \
    ln -sf $(which python3) /usr/bin/python && \
    pip3 install --upgrade pip

# Install system dependencies
RUN apt-get update && \
    apt-get install -y fonts-cantarell fontconfig git icu-devtools ipython3 libcairo2-dev libhdf5-dev pkg-config ttf-ubuntu-font-family unzip wget

# Install all python requirements
COPY requirements.txt .

RUN pip3 install torch
RUN pip3 install -r requirements.txt

## Install all Google fonts and extract their metadata
RUN wget -O google_fonts.zip https://github.com/google/fonts/archive/${GOOGLE_FONTS_COMMIT}.zip && \
    unzip google_fonts.zip && \
    mkdir -p /usr/share/fonts/truetype/google-fonts && \
    find fonts-${GOOGLE_FONTS_COMMIT} -type f -name "*.ttf" | xargs -I{} sh -c "install -Dm644 {} /usr/share/fonts/truetype/google-fonts" && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Cantarell-*.ttf" -delete && \
    find /usr/share/fonts/truetype/google-fonts -type f -name "Ubuntu-*.ttf" -delete && \
    apt-get --purge remove fonts-roboto && \
    fc-cache -f > /dev/null && \
    find fonts-${GOOGLE_FONTS_COMMIT} -name "METADATA.pb" | xargs -I{} bash -c "dirname {} | cut -d'/' -f3 | xargs printf; printf ","; grep -i 'subset' {} | cut -d':' -f2 | paste -sd "," - | sed 's/\"//g'" > /usr/share/fonts/truetype/google-fonts/google_fonts_metadata

# Summarize all font licenses
RUN sh -c "echo \"license,font\"; find fonts-$GOOGLE_FONTS_COMMIT -name \"METADATA.pb\" | xargs -I{} bash -c \"dirname {} | cut -d'/' -f2,3 | sed -r 's/[/]+/,/g'\"" > font_licenses.csv

ENV PYTHONPATH "${PYTHONPATH}:/generator"
