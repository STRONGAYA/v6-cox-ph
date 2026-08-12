"""
Pytest configuration and fixtures for v6-cox-ph tests.
"""

import os
import pytest
import docker
import time
import json
import platform
import subprocess

from vantage6.client import UserClient as Client
from pathlib import Path


@pytest.fixture(scope="session")
def docker_client():
    """Get the Docker client and determine Docker host for the entire test session."""
    try:
        client = docker.from_env()
        client.ping()

        # Determine the appropriate Docker host
        docker_host = get_docker_host()
        print(f"Detected Docker host: {docker_host}")

        # Add the docker_host as an attribute to the client for easy access
        client.docker_host = docker_host

        return client
    except docker.errors.DockerException:
        pytest.skip("Docker not available")


def get_docker_host():
    """Get the Docker host based on the operating system."""
    system = platform.system().lower()
    
    if system == "linux":
        # On Linux, we need to get the bridge IP
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", "bridge", "--format", 
                 "{{range .IPAM.Config}}{{.Subnet}}{{end}}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                subnet = result.stdout.strip()
                # Extract the first IP from the subnet (e.g., 172.17.0.0/16 -> 172.17.0.1)
                parts = subnet.split('.')
                if len(parts) >= 3:
                    return f"{parts[0]}.{parts[1]}.{parts[2]}.1"
        except Exception:
            pass
        
        # Fallback for Linux
        return "172.17.0.1"
    else:
        # macOS and Windows
        return "host.docker.internal"


@pytest.fixture(scope="session")
def algorithm_image(docker_client):
    """Build the algorithm Docker image for the entire test session."""
    # Get repository root and derive package name from folder
    repo_root = Path(__file__).parent.parent
    pkg_name = "v6-cox-ph"

    # Create image tag from package name
    image_tag = f"{pkg_name}:ci-test"

    try:
        print(f"Building algorithm image from {repo_root}...")
        print(f"Package name: {pkg_name}")
        print(f"Image tag: {image_tag}")

        # Build Docker image for the algorithm
        build_result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                image_tag,
                "--build-arg",
                f"PKG_NAME={pkg_name}",
                str(repo_root),
            ],
            check=True,
            timeout=300,
            capture_output=True,
            text=True,
        )

        if build_result.returncode != 0:
            pytest.skip(
                f"Algorithm image build failed with exit code {build_result.returncode}:\n"
                f"STDOUT: {build_result.stdout}\n"
                f"STDERR: {build_result.stderr}"
            )

        # Verify the image was created
        try:
            inspect_result = subprocess.run(
                ["docker", "inspect", image_tag],
                check=True,
                capture_output=True,
                text=True,
            )

            image_info = json.loads(inspect_result.stdout)
            if not image_info or not image_info[0].get("Id"):
                pytest.skip(f"Built image {image_tag} has no valid ID")

            print(f"Successfully built algorithm image: {image_tag}")
            return {
                "tag": image_tag,
                "pkg_name": pkg_name,
                "id": image_info[0]["Id"],
                "info": image_info[0],
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            pytest.skip(f"Failed to inspect built image {image_tag}: {e}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Algorithm build failed: {e}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        pytest.skip(error_msg)
    except subprocess.TimeoutExpired:
        pytest.skip("Timeout building algorithm image (300s)")
    except Exception as e:
        pytest.skip(f"Unexpected error building algorithm image: {e}")


@pytest.fixture(scope="session")
def extra_node_config_file(algorithm_image):
    """Create a temporary YAML file with the allowed algorithms policy."""
    algorithm_name = algorithm_image["tag"]

    # Retrieve the node configuration
    tests_directory = Path(__file__).parent
    config_file_path = os.path.join(
        tests_directory, "data", "additional_vantage6_node_config.yaml"
    )

    # Read existing content before modification
    original_content = ""
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            original_content = f.read()

    # Append new content
    new_content = (
        f"\n"
        f"policies:\n"
        f"  allowed_algorithms:\n"
        f"    # Developer network start-up test\n"
        f"    - ^hello-world$\n"
        f"    # {algorithm_name}\n"
        f"    - ^{algorithm_name}$\n"
        f"  require_algorithm_pull: false\n"
    )

    with open(config_file_path, "a") as config_file:
        config_file.write(new_content)

    # Use the absolute path to ensure no issues
    full_path = os.path.abspath(config_file_path)

    print(f"Updated extra node config file: {full_path}")
    print(f"Allowing algorithm: {algorithm_name}")

    yield full_path

    # Restore original content (removing what we added)
    try:
        if original_content:
            # Restore to its original state
            with open(config_file_path, "w") as f:
                f.write(original_content)
            print(f"Restored original content of config file: {full_path}")
        else:
            # File didn't exist before, strip the content we added
            with open(config_file_path, "r") as f:
                current_content = f.read()
            # Remove the new content we appended
            restored_content = current_content.replace(new_content, "")
            with open(config_file_path, "w") as f:
                f.write(restored_content)
            print(f"Stripped added content from config file: {full_path}")
    except OSError as e:
        print(f"Warning: Could not restore config file {full_path}: {e}")


@pytest.fixture(scope="session")
def vantage6_network_session(docker_client, extra_node_config_file):
    """Set up the Vantage6 developer network for the entire test session."""
    from tests.integration.test_vantage6_integration import cleanup_vantage6_network

    network_info = {"status": "not_started", "created_containers": set()}

    # Check if vantage6 CLI is available
    try:
        result = subprocess.run(
            ["v6", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            pytest.skip(
                f"Vantage6 CLI not available (exit code {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        pytest.skip("Vantage6 CLI check timed out (10s)")
    except FileNotFoundError:
        pytest.skip(
            "Vantage6 CLI not found in PATH. Install with: pip install vantage6"
        )

    try:
        # Clean-up of any existing network first
        print("Cleaning up any existing Vantage6 networks...")
        cleanup_vantage6_network(
            {"created_containers": set()}, docker_client, force_remove_existing=True
        )
        time.sleep(5)

        # Capture containers before creating the network
        containers_before = set(
            container.id for container in docker_client.containers.list(all=True)
        )

        # Get test data directory
        tests_directory = Path(__file__).parent
        data_directory = tests_directory / "data"

        # Define config file paths
        server_config_path = data_directory / "additional_vantage6_server_config.yaml"
        store_config_path = data_directory / "additional_vantage6_store_config.yaml"

        # Define dataset paths and names
        datasets = [
            ("coxph_test_data_1", data_directory / "coxph_test_data_1.csv"),
            ("coxph_test_data_2", data_directory / "coxph_test_data_2.csv"),
            ("coxph_test_data_3", data_directory / "coxph_test_data_3.csv"),
        ]

        # Create and start a demo network with extra node config
        print("Creating demo network: algorithm-ci-test.")
        print(f"Using IP: '{docker_client.docker_host}', as server IP.")
        create_args = [
            "v6",
            "dev",
            "create-demo-network",
            "--name",
            "algorithm-ci-test",
            "--server-url",
            str(docker_client.docker_host),
            "--extra-node-config",
            str(extra_node_config_file),
            "--extra-server-config",
            str(server_config_path),
            "--extra-store-config",
            str(store_config_path),
        ]

        # Add datasets dynamically
        for dataset_name, dataset_path in datasets:
            if dataset_path.exists():
                create_args.extend(["--add-dataset", dataset_name, str(dataset_path)])
            else:
                print(f"Warning: Dataset file {dataset_path} not found, skipping...")

        create_result = subprocess.run(
            create_args, timeout=300, capture_output=True, text=True
        )

        if create_result.returncode != 0:
            error_msg = f"Failed to create demo network (exit code {create_result.returncode}):\n"
            error_msg += f"STDOUT: {create_result.stdout}\n"
            error_msg += f"STDERR: {create_result.stderr}"
            pytest.skip(error_msg)

        print("Starting demo network...")
        start_result = subprocess.run(
            ["v6", "dev", "start-demo-network", "--name", "algorithm-ci-test"],
            timeout=300,
            capture_output=True,
            text=True,
        )

        if start_result.returncode != 0:
            error_msg = (
                f"Failed to start demo network (exit code {start_result.returncode}):\n"
            )
            error_msg += f"STDOUT: {start_result.stdout}\n"
            error_msg += f"STDERR: {start_result.stderr}"
            pytest.skip(error_msg)

        # Wait for the network to be ready
        print("Waiting for network to start...")
        max_wait = 90
        wait_interval = 5
        stable_count = 0
        required_stable_checks = 3

        for elapsed in range(0, max_wait, wait_interval):
            time.sleep(wait_interval)
            containers_after = set(
                container.id for container in docker_client.containers.list(all=True)
            )
            new_containers = containers_after - containers_before
            network_info["created_containers"] = new_containers

            # Check if we have enough containers running
            running_containers = docker_client.containers.list(
                filters={"status": "running"}
            )
            service_containers = [
                c for c in running_containers 
                if any(tag in c.name for tag in ["server", "node", "store"])
            ]

            if len(service_containers) >= 3:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    network_info["status"] = "running"
                    break
            else:
                stable_count = 0

            print(f"Waiting for network... ({elapsed}/{max_wait}s) "
                  f"Containers: {len(service_containers)} running")

        if network_info["status"] != "running":
            pytest.skip(
                f"Network did not start within {max_wait} seconds. "
                f"Running containers: {len(service_containers)}"
            )

        print(f"Network started successfully with {len(new_containers)} containers")

        yield network_info

        # Clean up at the end of the session
        print("Cleaning up Vantage6 network...")
        cleanup_vantage6_network(network_info, docker_client)

    except Exception as e:
        print(f"Error setting up Vantage6 network: {e}")
        # Try to clean up even if there was an error
        try:
            cleanup_vantage6_network(network_info, docker_client)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        pytest.skip(f"Failed to set up Vantage6 network: {e}")


@pytest.fixture(scope="session")
def authentication(vantage6_network_session):
    """Authenticate a vantage6 client with retry logic."""
    max_retries = 10
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            client = Client(
                "http://localhost",
                5000,
                "/api"
            )

            # Try to authenticate
            client.setup_encryption(None)
            client.authenticate(
                username="admin",
                password="admin"
            )

            # Test the connection
            organizations = client.organization.list()
            if organizations:
                print(f"Successfully authenticated. Found {len(organizations)} organizations.")
                return client
            else:
                print("Authentication succeeded but no organizations found.")

        except Exception as e:
            print(f"Authentication attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    pytest.skip("Failed to authenticate with Vantage6 server")


@pytest.fixture
def variables_config():
    """Configuration for test variables specific to Cox-PH algorithm."""
    return {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": ["age", "treatment"],
        "organization_ids": [1, 2, 3]
    }


@pytest.fixture
def coxph_test_data():
    """Load Cox-PH test data for unit testing."""
    import pandas as pd
    from pathlib import Path
    
    test_data_dir = Path(__file__).parent / "data"
    
    data_files = {
        "data_1": test_data_dir / "coxph_test_data_1.csv",
        "data_2": test_data_dir / "coxph_test_data_2.csv",
        "data_3": test_data_dir / "coxph_test_data_3.csv",
    }
    
    datasets = {}
    for name, path in data_files.items():
        if path.exists():
            datasets[name] = pd.read_csv(path)
    
    return datasets


@pytest.fixture
def coxph_variables():
    """Standard Cox-PH variable configuration for testing."""
    return {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": ["age", "treatment"],
    }
