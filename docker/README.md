# Docker container project for using [AugmentA](https://github.com/KIT-IBT/AugmentA)

For ARM this project only works with emulating AMD in the docker container (described below).
Dockerfile-x64 is currently the only working dockerfile.
If you have fun fixing the implementation for ARM MAC OS/AARCH  in the container go for it :) .
Work currently in progress

<h1 style="text-align: center; color: red; font-size: 20px;">Docker on MAC ARM PCs only works with MAC OS 14.6.1 or higher</h1>


# Installation

1. Download this repo
2. Install a VNC viewer of your choice (RealVNC Viewer, Native screen sharing app on MAC OS, ...)
3. Build Dockerfile-x64
    - ```docker build -f Dockerfile-x64 -t augmenta_container_image .```
    - For ARM PCs (e.g. M3 MAC):
      ```docker build --platform linux/amd64 -f Dockerfile-x64 -t augmenta_container_image .```
4. Change mounted folder to a folder on your pc ```"your/path/to/folder":headless/data``` in _Docker-compose.yml_
5. Start docker container with `docker-compose up`

# Headless (lightweight) image

If AugmentA does not need any user input (no point picking / GUI), use `Dockerfile-headless`.
It has no VNC/desktop, uses a multi-stage build (no compilers in the final image) and builds openCARP in Release mode.
All dependency versions are identical to `Dockerfile-x64`.

1. Build: ```docker build -f Dockerfile-headless -t augmenta_headless_image .```
    - For ARM PCs (e.g. M3 MAC): ```docker build --platform linux/amd64 -f Dockerfile-headless -t augmenta_headless_image .```
      (vtk 9.3.1, pymeshlab 2023.12.post2 and pymesh2 0.3.1 only exist for x86_64)
2. Change the mounted data folder in _Docker-compose-headless.yml_
3. Start an interactive shell: ```docker compose -f Docker-compose-headless.yml run --rm augmenta_headless```
   or run directly: ```docker compose -f Docker-compose-headless.yml run --rm augmenta_headless python main.py ...```

### Native ARM64 (Apple Silicon)

`Dockerfile-headless-arm64` runs natively on ARM without amd64 emulation (about 1.3-2.2x faster on an M3 Pro).
Because the pinned versions have no ARM64 builds, it uses vtk 9.5.2 and pymeshlab 2025.7 and compiles PyMesh from source;
all other versions are identical. Check your results against the x86 image if exact numerical reproducibility matters.

* Build: ```docker build -f Dockerfile-headless-arm64 -t augmenta_headless_arm64_image .```
* Run: ```docker compose -f Docker-compose-headless.yml run --rm augmenta_headless_arm64```

Notes:
* Matplotlib uses the `Agg` backend; interactive VTK windows are not available.

# Running AugmentA

## Access to UI

Because AugmentA in most cases needs user input at some point you have to get access to the UI of the docker container.
Therfore, you have to follow these steps:

### VNC viewer/screen sharing

1. Connect with your chosen VNC app to the following address localhost:5901 (localhost:5901/?password if you can not
   directly type in a password)
2. The password is vncpassword

### Browser based access

1. Open localhost:6901/?password in your browser
2. The password is vncpassword

## Storage

To mount your data, modify the volumes entry in [Docker-compose.yml](src/Docker-compose.yml)

## All in docker

1. Git clone AugmtentA from repo [https://github.com/KIT-IBT/AugmentA](https://github.com/KIT-IBT/AugmentA)
2. Run main.py
3. Further instructions see the tutorial in the AugmentA repo

## IDE based deployment (pycharm/intellij)

Developing AugmentA can be quite difficult when only using docker.
In combination with an IDE of your choice (we will demonstrate it for PyCharm/InteliJ **Professional**)
you can do debugging from your main pc within a docker container. Here is a quick guide how to set things up:

1. Install docker/docker desktop on your pc and start the docker engine
2. Install the docker plugin in PyCharm Settings/Plugins
3. Under Build,execution,Deployment/Docker create a new instance of docker which connects to the default port
4. In your project settings/Python Interpreter create a new interpreter with `On Docker Compose ...`
5. Select here `.\src\Docker-compose.yml` as the configuration file and `augmenta_debugger` as the selected service

## Tipps
* AugmentA in docker works currently best with absolute paths starting with `/headless/data`
* The container does not include AugmentA itself, download it here [https://github.com/KIT-IBT/AugmentA](https://github.com/KIT-IBT/AugmentA)