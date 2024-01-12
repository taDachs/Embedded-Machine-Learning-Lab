# Ubuntu 18.04
FROM ubuntu:bionic

ARG USERNAME=eml
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip git git-lfs python3-venv python3-tk

RUN pip3 install --upgrade pip && \
  pip3 install virtualenv pre-commit tldr python-lsp-server[all] pylint black-21.12b0


USER $USERNAME
CMD ["/bin/bash"]
