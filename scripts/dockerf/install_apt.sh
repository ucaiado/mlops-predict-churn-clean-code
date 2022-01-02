apt-get update -qq
apt-get install -y \
        apt-utils \
        apt-transport-https \
        dirmngr \
        gnupg \
        libcurl4-openssl-dev \
        libnlopt-dev \
        lsb-release \
        libssl-dev \
        libgsl-dev
apt-key adv \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
add-apt-repository \
          --yes \
          ppa:c2d4u.team/c2d4u4.0+

apt-get install -y \
        aptdaemon \
        ed \
        git \
        mercurial \
        libcairo-dev \
        libedit-dev \
        libxml2-dev \
        tmux \
        sudo \
        wget \
        build-essential

rm -rf /var/lib/apt/lists/*
