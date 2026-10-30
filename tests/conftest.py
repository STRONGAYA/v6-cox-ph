"""
Pytest configuration and fixtures for a vantage6 algorithm's integration
tests. Copied from this repository's `test-infra/` into the consumer's
`tests/conftest.py`; a consumer changes ONLY the two marked spots: the
`DATASETS` constant below and the `variables_config` fixture at the end.
Fixes to everything else go to the conventions repository, not to a local
copy.
"""

import os
import pytest
import docker
import time
import json
import platform
import re
import subprocess
import requests

from vantage6.client import UserClient as Client
from pathlib import Path

# The STRONG AYA RDF test store: a disposable GraphDB holding the
# consumer's tests/data/rdf/ turtle files, for consumers that test RDF
# data access. The image is digest-pinned; the container name and the
# host port belong to the test suite alone — never the developer's own
# Flyover store (which listens on 7200).
RDF_STORE_IMAGE = (
    "ghcr.io/maastrichtu-cds/flyover-graphdb" "@sha256:31ce8d36ba9b6bc8b2ed59732dc403f8d4ee8bc818de860d79029980b01fb614"
)
RDF_STORE_CONTAINER = "v6-coxph-rdf-test"
RDF_STORE_PORT = 7201
RDF_STORE_REPOSITORY = "userRepo"

# Environment variables that indicate a continuous integration environment;
# 'CI' is set by GitHub Actions and most other providers.
CI_ENVIRONMENT_VARIABLES = ("CI", "REQUIRE_INTEGRATION_TESTS")


def _running_in_ci():
    """Determine whether the tests are run in a CI environment."""
    return any(
        os.environ.get(variable, "").strip().lower() in ("1", "true", "yes") for variable in CI_ENVIRONMENT_VARIABLES
    )


def _skip_or_fail(reason):
    """Skip when run locally, but fail in a continuous integration environment.

    Infrastructure that cannot be set up on a developer's machine — Docker or
    the vantage6 CLI that is not installed, for instance — is a reason to skip
    the integration tests. The same situation in continuous integration means
    that the integration tests verify nothing whilst the run still reports
    success, which is why it is an error there.
    """
    if _running_in_ci():
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


# EDIT POINT 1 of 2 — the datasets the dev network attaches to the nodes.
# One (label, path) pair per dataset; the CLI splits each file across the
# nodes in contiguous slices, so a multi-node test needs one file with equal
# blocks per node (see the audit harness's canary.py for the layout).
DATASETS = [
    ("coxph_test_data_1", Path(__file__).parent / "data" / "coxph_test_data_1.csv"),
    ("coxph_test_data_2", Path(__file__).parent / "data" / "coxph_test_data_2.csv"),
    ("coxph_test_data_3", Path(__file__).parent / "data" / "coxph_test_data_3.csv"),
]


@pytest.fixture(scope="session")
def docker_test_setup():
    """Check if Docker is available for testing."""
    try:
        client = docker.from_env()
        client.ping()
        return {"docker_available": True, "client": client}
    except docker.errors.DockerException:
        return {"docker_available": False, "client": None}


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
        _skip_or_fail("Docker not available")


def _setuptools_packages(pyproject_text):
    """Return the ``packages`` list of ``[tool.setuptools]``, or ``None``.

    ``tomllib`` is used when it is available (Python 3.11 and later); on
    older interpreters a minimal parse of the ``packages`` value inside the
    ``[tool.setuptools]`` section is the fallback — the test infrastructure
    only needs a plain list of package names.
    """
    try:
        import tomllib
    except ImportError:  # Python 3.10 has no tomllib
        tomllib = None

    if tomllib is not None:
        try:
            data = tomllib.loads(pyproject_text)
        except tomllib.TOMLDecodeError:
            return None
        packages = data.get("tool", {}).get("setuptools", {}).get("packages")
        if isinstance(packages, list) and packages:
            return [str(package) for package in packages]
        return None

    lines = pyproject_text.splitlines()
    in_section = False
    position = 0
    while position < len(lines):
        stripped = lines[position].strip()
        position += 1
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == "[tool.setuptools]"
            continue
        if not in_section:
            continue
        match = re.match(r"packages\s*=\s*\[(.*)$", stripped)
        if not match:
            continue
        # The list may span several lines; collect up to the closing bracket.
        collected = match.group(1)
        while "]" not in collected and position < len(lines):
            collected += lines[position].strip()
            position += 1
        inner = collected.split("]", 1)[0]
        packages = [item.strip().strip("\"'") for item in inner.split(",")]
        packages = [package for package in packages if package]
        return packages or None
    return None


def _dockerfile_pkg_name(repo_root):
    """Return the ``ARG PKG_NAME`` default of the Dockerfile, or ``None``."""
    try:
        lines = (repo_root / "Dockerfile").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        match = re.match(r"\s*ARG\s+PKG_NAME\s*=\s*(.+?)\s*$", line)
        if match:
            value = match.group(1).strip().strip("\"'")
            if value:
                return value
    return None


def _package_name_from_project(repo_root):
    """Derive the algorithm package name from the project itself.

    The package name must not come from the checkout folder's name: a
    consumer repository checked out under a different folder (a GitHub
    Actions checkout uses the repository name, for instance) would otherwise
    build an image that imports a package that does not exist. The name is
    read from ``[tool.setuptools] packages[0]`` in ``pyproject.toml``; the
    Dockerfile's ``ARG PKG_NAME`` is the fallback when the pyproject cannot
    be read or parsed.
    """
    try:
        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        pyproject_text = None

    if pyproject_text is not None:
        packages = _setuptools_packages(pyproject_text)
        if packages:
            return packages[0]

    dockerfile_name = _dockerfile_pkg_name(repo_root)
    if dockerfile_name:
        return dockerfile_name

    _skip_or_fail(
        "Cannot determine the algorithm package name: no readable "
        "[tool.setuptools] packages in pyproject.toml and no ARG PKG_NAME "
        "in the Dockerfile"
    )


@pytest.fixture(scope="session")
def algorithm_image(docker_client):
    """Build the algorithm Docker image for the entire test session."""
    # Get repository root and derive the package name from the project
    # (pyproject.toml [tool.setuptools] packages[0], falling back to the
    # Dockerfile's ARG PKG_NAME) — never from the checkout folder's name.
    repo_root = Path(__file__).parent.parent
    pkg_name = _package_name_from_project(repo_root)

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
            _skip_or_fail(
                f"Algorithm image build failed with exit code "
                f"{build_result.returncode}:\n"
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
                _skip_or_fail(f"Built image {image_tag} has no valid ID")

            print(f"Successfully built algorithm image: {image_tag}")
            return {
                "tag": image_tag,
                "pkg_name": pkg_name,
                "id": image_info[0]["Id"],
                "info": image_info[0],
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            _skip_or_fail(f"Failed to inspect built image {image_tag}: {e}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Algorithm build failed: {e}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        _skip_or_fail(error_msg)
    except subprocess.TimeoutExpired:
        _skip_or_fail("Timeout building algorithm image (300s)")
    except Exception as e:
        _skip_or_fail(f"Unexpected error building algorithm image: {e}")


@pytest.fixture(scope="session")
def extra_node_config_file(algorithm_image):
    """Create a temporary YAML file with the allowed algorithms policy."""
    algorithm_name = algorithm_image["tag"]

    # Retrieve the node configuration
    tests_directory = Path(__file__).parent
    config_file_path = os.path.join(tests_directory, "data", "additional_vantage6_node_config.yaml")

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
def rdf_store(docker_client):
    """Start a disposable GraphDB holding the STRONG AYA RDF test data.

    Optional: a consumer without RDF test data — any of the turtle files
    under ``tests/data/rdf/`` missing — gets ``None`` and nothing is
    started, so CSV-only consumers are unaffected. The container is
    removed at teardown only when this fixture started it; a developer's
    own store is never stopped or removed.
    """
    rdf_directory = Path(__file__).parent / "data" / "rdf"
    if not all((rdf_directory / f"{name}.ttl").is_file() for name in ("ontology", "data", "annotation")):
        yield None
        return

    base_url = f"http://localhost:{RDF_STORE_PORT}"
    started = False
    try:
        container = docker_client.containers.get(RDF_STORE_CONTAINER)
        if container.status != "running":
            container.start()
    except docker.errors.NotFound:
        try:
            container = docker_client.containers.run(
                RDF_STORE_IMAGE,
                name=RDF_STORE_CONTAINER,
                detach=True,
                ports={"7200/tcp": ("127.0.0.1", RDF_STORE_PORT)},
                environment={"GDB_HEAP_SIZE": "2g"},
            )
        except docker.errors.DockerException as e:
            _skip_or_fail(f"Could not start the RDF store: {e}")
        started = True
    try:
        deadline = time.time() + 240
        while True:
            try:
                if requests.get(f"{base_url}/rest/repositories", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            if time.time() > deadline:
                _skip_or_fail("The RDF store did not start within 240 seconds")
            time.sleep(5)
        repository = f"{base_url}/repositories/{RDF_STORE_REPOSITORY}"
        if int(requests.get(f"{repository}/size", timeout=60).text) == 0:
            for name in ("ontology", "data", "annotation"):
                with open(rdf_directory / f"{name}.ttl", "rb") as turtle:
                    response = requests.post(
                        f"{repository}/rdf-graphs/service?graph=http://{name}.local/",
                        data=turtle.read(),
                        headers={"Content-Type": "application/x-turtle"},
                        timeout=300,
                    )
                if not response.ok:
                    _skip_or_fail(f"Could not load {name}.ttl into the RDF store: {response.status_code}")
        yield {"endpoint": repository, "container": RDF_STORE_CONTAINER}
    finally:
        if started:
            container.remove(force=True)


@pytest.fixture(scope="session")
def vantage6_network_session(docker_client, extra_node_config_file, rdf_store):
    """Set up the Vantage6 developer network for the entire test session."""
    from tests.integration.test_vantage6_integration import cleanup_vantage6_network

    # The RDF store must be up before the network starts: the node links
    # the container at start-up through docker_services. A consumer
    # without RDF test data gets None here and starts nothing.
    if rdf_store is not None:
        print(f"RDF store available at {rdf_store['endpoint']}")

    network_info = {"status": "not_started", "created_containers": set()}

    # Check if vantage6 CLI is available
    try:
        result = subprocess.run(["v6", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            _skip_or_fail(
                f"Vantage6 CLI not available (exit code {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        _skip_or_fail("Vantage6 CLI check timed out (10s)")
    except FileNotFoundError:
        _skip_or_fail("Vantage6 CLI not found in PATH. Install with: pip install vantage6")

    try:
        # Clean-up of any existing network first
        print("Cleaning up any existing Vantage6 networks...")
        cleanup_vantage6_network({"created_containers": set()}, docker_client, force_remove_existing=True)
        time.sleep(5)

        # Capture containers before creating the network
        containers_before = set(container.id for container in docker_client.containers.list(all=True))

        # Get test data directory
        tests_directory = Path(__file__).parent
        data_directory = tests_directory / "data"

        # Define config file paths
        server_config_path = data_directory / "additional_vantage6_server_config.yaml"
        store_config_path = data_directory / "additional_vantage6_store_config.yaml"

        # The RDF dataset is appended here rather than in DATASETS (the
        # consumer's edit point); the missing-file warning below already
        # skips consumers without rdf_store.csv.
        datasets = DATASETS + [("rdf_store", data_directory / "rdf_store.csv")]

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

        create_result = subprocess.run(create_args, timeout=300, capture_output=True, text=True)

        if create_result.returncode != 0:
            error_msg = f"Failed to create demo network " f"(exit code {create_result.returncode}):\n"
            error_msg += f"STDOUT: {create_result.stdout}\n"
            error_msg += f"STDERR: {create_result.stderr}"
            _skip_or_fail(error_msg)

        print("Starting demo network...")
        start_result = subprocess.run(
            ["v6", "dev", "start-demo-network", "--name", "algorithm-ci-test"],
            timeout=300,
            capture_output=True,
            text=True,
        )

        if start_result.returncode != 0:
            error_msg = f"Failed to start demo network " f"(exit code {start_result.returncode}):\n"
            error_msg += f"STDOUT: {start_result.stdout}\n"
            error_msg += f"STDERR: {start_result.stderr}"
            _skip_or_fail(error_msg)

        # Wait for the network to be ready
        print("Waiting for network to start...")
        max_wait = 90
        wait_interval = 5
        stable_count = 0
        required_stable_checks = 3

        for elapsed in range(0, max_wait, wait_interval):
            time.sleep(wait_interval)
            containers_after = set(container.id for container in docker_client.containers.list(all=True))
            new_containers = containers_after - containers_before

            if len(new_containers) >= 4:
                service_containers = 0
                for container_id in new_containers:
                    try:
                        container = docker_client.containers.get(container_id)
                        if (
                            container.status == "running"
                            and "-run-" not in container.name
                            and "algorithm-store" not in container.name
                        ):
                            service_containers += 1
                    except docker.errors.NotFound:
                        pass

                if service_containers >= 3:
                    stable_count += 1
                    if stable_count >= required_stable_checks:
                        print(f"Network stable and ready after " f"{elapsed + wait_interval} seconds")
                        break
                else:
                    stable_count = 0

        containers_after = set(container.id for container in docker_client.containers.list(all=True))
        network_info["created_containers"] = containers_after - containers_before
        network_info["status"] = "running"

        if len(network_info["created_containers"]) < 4:
            _skip_or_fail(
                "Network setup incomplete: "
                f"only {len(network_info['created_containers'])} "
                "containers created (expected at least 4)"
            )

        print(f"Network started with " f"{len(network_info['created_containers'])} new containers")

        yield network_info

    except subprocess.TimeoutExpired as e:
        error_msg = f"Timeout setting up Vantage6 demo network: {e}"
        _skip_or_fail(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error setting up Vantage6 demo network: {e}"
        _skip_or_fail(error_msg)

    finally:
        # Clean-up after all tests are done
        print("Cleaning up Vantage6 network...")
        cleanup_vantage6_network(network_info, docker_client)


@pytest.fixture(scope="session")
def authentication(vantage6_network_session, docker_client) -> Client:
    """
    Authenticate a vantage6 client.

    Creates a client, authenticates it, and sets up encryption.
    """
    # Ensure the network is running before attempting authentication
    if vantage6_network_session["status"] != "running":
        _skip_or_fail("Vantage6 network is not running - cannot authenticate")

    # Additional wait to ensure all services are fully ready
    print("Waiting for network services to be fully ready for authentication...")
    time.sleep(10)

    vantage6_config = {
        "server_url": "http://localhost",
        "server_port": 7601,
        "server_api": "/api",
        "username": "dev_admin",
        "password": "password",
    }

    # Create a client
    vantage6_client = Client(
        vantage6_config.get("server_url"),
        vantage6_config.get("server_port"),
        vantage6_config.get("server_api"),
        log_level="debug",
    )

    # Authenticate the client with a retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vantage6_client.authenticate(
                vantage6_config.get("username"),
                vantage6_config.get("password"),
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                _skip_or_fail(f"Failed to authenticate after {max_retries} attempts: {e}")
            print(f"Authentication attempt {attempt + 1} failed, " f"retrying... Error: {e}")
            time.sleep(5)

    # Verify authentication
    assert hasattr(vantage6_client, "token"), "Client should have a token attribute"
    assert vantage6_client.token, "Authentication token should be present and not empty"

    return vantage6_client


@pytest.fixture(scope="session")
def algorithm_image_name(algorithm_image):
    """Get the algorithm image name for testing."""
    return algorithm_image["tag"]


def get_docker_host():
    """
    Determine the appropriate Docker host address based on the OS.

    Returns
    -------
    str
        The Docker host address to use.
    """
    system = platform.system().lower()

    if system in ["windows", "darwin"]:  # Windows or macOS
        return "http://host.docker.internal"
    else:  # Linux and others
        try:
            # Try to get the Docker bridge IP
            result = subprocess.run(
                ["docker", "network", "inspect", "bridge"],
                capture_output=True,
                text=True,
                check=True,
            )
            bridge_info = json.loads(result.stdout)
            gateway = bridge_info[0]["IPAM"]["Config"][0]["Gateway"]

            # The gateway might not have the http scheme included
            if not gateway.startswith("http"):
                gateway = f"http://{gateway}"
            return gateway
        except (
            subprocess.CalledProcessError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ):
            # Assume it is a default Docker IP if inspection fails
            return "http://172.17.0.1"


# Pytest markers for organising tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for algorithm functions")
    config.addinivalue_line("markers", "integration: Integration tests with Vantage6")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "vantage6: Tests that require Vantage6 infrastructure")
    config.addinivalue_line("markers", "docker: Tests that require Docker")


def pytest_collection_modifyitems(config, items):
    """Modify the test collection to add markers based on test names/paths."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Add a vantage6 marker to tests that use vantage6
        if "vantage6" in item.nodeid or "vantage6" in item.name:
            item.add_marker(pytest.mark.vantage6)

        # Add docker marker to tests that use docker
        if "docker" in item.nodeid or "docker" in item.name:
            item.add_marker(pytest.mark.docker)


@pytest.fixture
def variables_config():
    """Fixture providing test variable configuration for the Cox-PH algorithm."""
    return {
        "time_col": "time",
        "outcome_col": "event",
        "expl_vars": ["age", "treatment"],
        "organization_ids": [1, 2, 3],
    }
