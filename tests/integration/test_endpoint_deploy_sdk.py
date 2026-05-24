import pytest

pytestmark = pytest.mark.integration


def test_deploy_and_smoke_test_real_workspace():
    """Real workspace integration test.

    Requires:
        - DATABRICKS_HOST + DATABRICKS_TOKEN env vars
        - A registered UC model with @candidate alias set
        - Permission to create serving endpoints
    """
    pytest.skip("Manual integration test; run only against a real workspace with credentials")
