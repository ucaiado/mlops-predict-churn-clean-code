FROM continuumio/miniconda3:4.9.2

COPY install_apt.sh /tmp/
RUN \
  sh /tmp/install_apt.sh \
  && rm -rf /tmp/* \
  && apt-get remove --purge -y $BUILDDEPS \
  && apt-get autoremove -y \
  && apt-get autoclean -y \
  && rm -rf /var/lib/apt/lists/*


# Install additional python packages
COPY requirements.txt /tmp/
RUN pip install Cython --install-option="--no-cython-compile"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /root/.cache

RUN conda install scikit-learn==0.22 -y --quiet && \
    pip install --no-cache-dir shap && \
    rm -rf /root/.cache

# Setup jupyter
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension install --py jupytext --user && \
    jupyter nbextension enable --py jupytext --user
