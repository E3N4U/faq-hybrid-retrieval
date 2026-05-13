import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tests.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_calculator_data():
    return {
        'operation': 'add',
        'a': 10,
        'b': 5
    }


@pytest.fixture
def sample_string_data():
    return {
        'text': 'Hello World',
        'operation': 'upper'
    }
