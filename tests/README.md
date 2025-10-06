# Testing Setup for AugmentA
This document describes how to run tests for the AugmentA project using Docker Compose.

## Overview
The testing environment for AugmentA uses Docker Compose to manage the test container. This approach ensures proper test isolation, environment consistency, and provides explicit control over the test container lifecycle.

## Prerequisites
- Docker and Docker Compose installed on your system
- Access to the `augmentadocker/src` directory

## Running Tests
### Step 1: Start the Docker Compose Services

Open a terminal and navigate to the source directory:
```bash
cd augmentadocker/src
```

Start the Docker Compose services:
```bash
docker-compose up
```
**What this does**: This command starts all services defined in the `docker-compose.yml` file, including the `augmenta_debugger` container. Keep this terminal window open - it will display logs from the running services.

### Step 2: Access the Test Container
Open a **second terminal window** and navigate to the same directory:
```bash
cd augmentadocker/src
```

Execute a bash shell inside the running test container:
```bash
docker-compose exec augmenta_debugger bash
```
**What this does**: This connects you to an interactive bash session inside the `augmenta_debugger` container where the test environment is configured.

### Step 3: Run the Tests
Inside the container shell, run pytest with verbose and output options:
```bash
pytest -v -s
```
**Command options**:
- `-v` (verbose): Shows detailed test output including individual test names
- `-s` (no capture): Displays print statements and logging output during test execution

## Copying Files from Container to Mac
If you need to inspect test results, debug outputs, or any files generated during testing, you can copy them from the container's temporary directory to your Mac desktop.

### Copy Files to Local Desktop While Container is Running
From your Mac terminal (not inside the container):
```bash
docker cp augmenta_debugger:/tmp/pytest-of-default/pytest-[number]/test_la_cut_resample_pipeline_[number]/input/LA_cutted_res_surf/ ~/Desktop/
```
**Example**: Copy the file called ring_3.vtk from the temp directory to desktop:
```bash
# This copies the only ring_3.vtk file to your Desktop
docker cp augmenta_debugger:/tmp/pytest-of-default/pytest-1/test_la_cut_resample_pipeline_0/input/LA_cutted_res_surf/ring_3.vtk ~/Desktop/
```
**Example**: Copy all program output from the temp directory:
```bash
# This copies the entire LA_cutted_res_surf directory to your Desktop
docker cp augmenta_debugger:/tmp/pytest-of-default/pytest-1/test_la_cut_resample_pipeline_0/input/LA_cutted_res_surf/ ~/Desktop/
```
**Note**: Replace `augmenta_debugger` with the actual container name if it differs. You can find the exact container name by running `docker ps`.

## Stopping the Test Environment
After testing is complete:
1. Exit the container shell by typing `exit` or pressing `cmd+D` or `ctrl+D`
2. In the first terminal window (where `docker-compose up` is running), press `cmd+C` or `crtl+C` to stop the services
3. Optionally, run `docker-compose down` to remove the containers

## Troubleshooting
### Tests not found or import errors
- Ensure you're inside the container (`docker-compose exec augmenta_debugger bash`) before running pytest
- Verify that the `docker-compose.yml` file correctly mounts the source code directory

### Container won't start
- Check that no other services are using the same ports
- Review the logs in the `docker-compose up` terminal for error messages
- Try `docker-compose down` followed by `docker-compose up` to restart clean

### Changes to code not reflected in tests
- The source code should be mounted as a volume in the container
- If changes aren't appearing, check the volume mappings in `docker-compose.yml`

### Cannot copy files from container
- Ensure the container is running (`docker ps` should show `augmenta_debugger`)
- Verify the source path exists inside the container using `docker-compose exec augmenta_debugger ls -la /tmp`
- Check that you have write permissions on the destination directory on your Mac

