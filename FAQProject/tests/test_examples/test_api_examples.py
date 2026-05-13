import pytest


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'test-demo-api'

    def test_health_endpoint_returns_json(self, client):
        response = client.get('/health')
        assert response.content_type == 'application/json'


class TestCalculatorEndpoint:
    def test_add_operation(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'add', 'a': 10, 'b': 5})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 15
        assert data['operation'] == 'add'

    def test_subtract_operation(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'subtract', 'a': 10, 'b': 3})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 7

    def test_multiply_operation(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'multiply', 'a': 4, 'b': 5})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 20

    def test_divide_operation(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'divide', 'a': 10, 'b': 2})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 5

    def test_divide_by_zero(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'divide', 'a': 10, 'b': 0})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_invalid_operation(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'invalid', 'a': 10, 'b': 5})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_missing_parameters(self, client):
        response = client.post('/api/calculator',
                                json={'operation': 'add'})
        assert response.status_code == 200


class TestStringOpsEndpoint:
    def test_upper_operation(self, client):
        response = client.post('/api/string_ops',
                                json={'text': 'hello', 'operation': 'upper'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 'HELLO'

    def test_lower_operation(self, client):
        response = client.post('/api/string_ops',
                                json={'text': 'HELLO', 'operation': 'lower'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 'hello'

    def test_reverse_operation(self, client):
        response = client.post('/api/string_ops',
                                json={'text': 'hello', 'operation': 'reverse'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 'olleh'

    def test_length_operation(self, client):
        response = client.post('/api/string_ops',
                                json={'text': 'hello', 'operation': 'length'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['result'] == 5

    def test_invalid_string_operation(self, client):
        response = client.post('/api/string_ops',
                                json={'text': 'hello', 'operation': 'invalid'})
        assert response.status_code == 400


class TestValidationEndpoint:
    def test_validate_email_valid(self, client):
        response = client.post('/api/validate',
                                json={'value': 'test@example.com', 'type': 'email'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is True

    def test_validate_email_invalid(self, client):
        response = client.post('/api/validate',
                                json={'value': 'testexample.com', 'type': 'email'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is False

    def test_validate_number_valid(self, client):
        response = client.post('/api/validate',
                                json={'value': '123', 'type': 'number'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is True

    def test_validate_positive_valid(self, client):
        response = client.post('/api/validate',
                                json={'value': '100', 'type': 'positive'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is True

    def test_validate_positive_invalid(self, client):
        response = client.post('/api/validate',
                                json={'value': '-50', 'type': 'positive'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is False

    def test_validate_string(self, client):
        response = client.post('/api/validate',
                                json={'value': 'hello', 'type': 'string'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['is_valid'] is True

    def test_validate_invalid_type(self, client):
        response = client.post('/api/validate',
                                json={'value': 'test', 'type': 'invalid'})
        assert response.status_code == 400
