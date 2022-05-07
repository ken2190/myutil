# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pytest

from recommenders.tuning.nni.nni_utils import (
    get_experiment_status,
    check_experiment_status,
    check_stopped,
    check_metrics_written,
    get_trials,
    NNI_STATUS_URL,
    NNI_TRIAL_JOBS_URL,
)


class MockResponse:
    # Class that mocks requests.models.Response
    def __init__(self, content, error):
        """ MockResponse:__init__.
        Doc::
                
                    Args:
                        content:     
                        error:     
                    Returns:
                       
        """
        self._content = content
        self._error = error

    def json(self):
        """ MockResponse:json.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        return {"status": self._content, "errors": [self._error]}


def mocked_status_get(url, content, error):
    """function mocked_status_get.
    Doc::
            
            Args:
                url:   
                content:   
                error:   
            Returns:
                
    """
    assert url.startswith(NNI_STATUS_URL)
    return MockResponse(content, error)


class MockResponseTrials:
    # Class that mocks requests.models.Response
    def __init__(self, content):
        """ MockResponseTrials:__init__.
        Doc::
                
                    Args:
                        content:     
                    Returns:
                       
        """
        self._content = content

    def json(self):
        """ MockResponseTrials:json.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        return self._content


def mocked_trials_get(url, content):
    """function mocked_trials_get.
    Doc::
            
            Args:
                url:   
                content:   
            Returns:
                
    """
    assert url.startswith(NNI_TRIAL_JOBS_URL)
    return MockResponseTrials(content)


def mock_exception():
    """function mock_exception.
    Doc::
            
            Args:
            Returns:
                
    """
    raise Exception()


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_get_experiment_status():
    """function test_get_experiment_status.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "some_status"
    error = ""
    with patch(
        "requests.get", side_effect=lambda url: mocked_status_get(url, content, error)
    ):
        nni_status = get_experiment_status(NNI_STATUS_URL)
        assert nni_status["status"] == "some_status"
        assert nni_status["errors"] == [""]


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_experiment_status_done():
    """function test_check_experiment_status_done.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "DONE"
    error = ""
    with patch(
        "requests.get", side_effect=lambda url: mocked_status_get(url, content, error)
    ):
        check_experiment_status(wait=0.1, max_retries=1)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_experiment_status_tuner_no_more_trial():
    """function test_check_experiment_status_tuner_no_more_trial.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "TUNER_NO_MORE_TRIAL"
    error = ""
    with patch(
        "requests.get", side_effect=lambda url: mocked_status_get(url, content, error)
    ):
        check_experiment_status(wait=0.1, max_retries=1)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_experiment_status_running():
    """function test_check_experiment_status_running.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "RUNNING"
    error = ""
    with pytest.raises(TimeoutError) as excinfo:
        with patch(
            "requests.get",
            side_effect=lambda url: mocked_status_get(url, content, error),
        ):
            check_experiment_status(wait=0.1, max_retries=1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_experiment_status_no_more_trial():
    """function test_check_experiment_status_no_more_trial.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "NO_MORE_TRIAL"
    error = ""
    with pytest.raises(TimeoutError) as excinfo:
        with patch(
            "requests.get",
            side_effect=lambda url: mocked_status_get(url, content, error),
        ):
            check_experiment_status(wait=0.1, max_retries=1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_experiment_status_failed():
    """function test_check_experiment_status_failed.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "some_failed_status"
    error = "NNI_ERROR"
    with pytest.raises(RuntimeError) as excinfo:
        with patch(
            "requests.get",
            side_effect=lambda url: mocked_status_get(url, content, error),
        ):
            check_experiment_status(wait=0.1, max_retries=1)
    assert (
        "NNI experiment failed to complete with status some_failed_status - NNI_ERROR"
        == str(excinfo.value)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_stopped_timeout():
    """function test_check_stopped_timeout.
    Doc::
            
            Args:
            Returns:
                
    """
    content = "some_status"
    error = ""
    with pytest.raises(TimeoutError) as excinfo:
        with patch(
            "requests.get",
            side_effect=lambda url: mocked_status_get(url, content, error),
        ):
            check_stopped(wait=0.1, max_retries=1)
    assert "check_stopped() timed out" == str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_stopped():
    """function test_check_stopped.
    Doc::
            
            Args:
            Returns:
                
    """
    with patch("requests.get", side_effect=mock_exception):
        check_stopped(wait=0.1, max_retries=1)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_metrics_written():
    """function test_check_metrics_written.
    Doc::
            
            Args:
            Returns:
                
    """
    content = [{"finalMetricData": None}, {"finalMetricData": None}]
    with patch("requests.get", side_effect=lambda url: mocked_trials_get(url, content)):
        check_metrics_written(wait=0.1, max_retries=1)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_check_metrics_written_timeout():
    """function test_check_metrics_written_timeout.
    Doc::
            
            Args:
            Returns:
                
    """
    content = [{"logPath": "/p"}, {"logPath": "/q"}]
    with pytest.raises(TimeoutError) as excinfo:
        with patch(
            "requests.get", side_effect=lambda url: mocked_trials_get(url, content)
        ):
            check_metrics_written(wait=0.1, max_retries=1)
    assert "check_metrics_written() timed out" == str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_get_trials():
    """function test_get_trials.
    Doc::
            
            Args:
            Returns:
                
    """
    with TemporaryDirectory() as tmp_dir1, TemporaryDirectory() as tmp_dir2:

        mock_trials = [
            {
                "finalMetricData": [
                    {"data": '"{\\"rmse\\": 0.8, \\"default\\": 0.3}"'}
                ],
                "logPath": "file://localhost:{}".format(tmp_dir1),
            },
            {
                "finalMetricData": [
                    {"data": '"{\\"rmse\\": 0.9, \\"default\\": 0.2}"'}
                ],
                "logPath": "file://localhost:{}".format(tmp_dir2),
            },
        ]
        metrics1 = {"rmse": 0.8, "precision_at_k": 0.3}
        with open(os.path.join(tmp_dir1, "metrics.json"), "w") as f:
            json.dump(metrics1, f)
        params1 = {
            "parameter_id": 1,
            "parameter_source": "algorithm",
            "parameters": {"n_factors": 100, "reg": 0.1},
        }
        with open(os.path.join(tmp_dir1, "parameter.cfg"), "w") as f:
            json.dump(params1, f)
        metrics2 = {"rmse": 0.9, "precision_at_k": 0.2}
        with open(os.path.join(tmp_dir2, "metrics.json"), "w") as f:
            json.dump(metrics2, f)
        params2 = {
            "parameter_id": 2,
            "parameter_source": "algorithm",
            "parameters": {"n_factors": 50, "reg": 0.02},
        }
        with open(os.path.join(tmp_dir2, "parameter.cfg"), "w") as f:
            json.dump(params2, f)

        with patch(
            "requests.get", side_effect=lambda url: mocked_trials_get(url, mock_trials)
        ):
            trials, best_metrics, best_params, best_trial_path = get_trials(
                optimize_mode="maximize"
            )

        expected_trials = [
            ({"rmse": 0.8, "default": 0.3}, tmp_dir1),
            ({"rmse": 0.9, "default": 0.2}, tmp_dir2),
        ]
        assert trials == expected_trials
        assert best_metrics == metrics1
        assert best_params == params1
        assert best_trial_path == tmp_dir1
