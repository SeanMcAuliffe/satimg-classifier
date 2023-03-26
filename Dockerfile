FROM tensorflow/tensorflow:latest-gpu

# Set user and group arguments
ARG USER_ID
ARG GROUP_ID

# Create a new user with the same UID and GID as the host user
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Grant user permissions to the Python package directory
RUN chown -R user:user /usr/local/lib/python3.8/dist-packages

# Install packages as root
USER root
RUN pip install --upgrade pandas matplotlib scikit-image

# Set the user for subsequent commands
USER user