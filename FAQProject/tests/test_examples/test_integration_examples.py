import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_examples.test_unit_examples import add, subtract, multiply, divide, validate_email


class TestIntegrationCalculator:
    def test_calculator_integration(self):
        result = add(10, 5)
        assert result == 15
        result = subtract(result, 3)
        assert result == 12
        result = multiply(result, 2)
        assert result == 24
        result = divide(result, 4)
        assert result == 6

    def test_calculator_chain_operations(self):
        numbers = [10, 5, 2]
        result = add(numbers[0], numbers[1])
        result = multiply(result, numbers[2])
        assert result == 30

    def test_email_validation_integration(self):
        test_cases = [
            ("user@example.com", True),
            ("test@domain.org", True),
            ("invalid-email", False),
            ("@nodomain.com", False),
            ("noatsign.com", False)
        ]
        for email, expected in test_cases:
            assert validate_email(email) == expected


class TestIntegrationMultiStepWorkflow:
    def test_order_processing_workflow(self):
        order = {
            'items': [{'price': 10}, {'price': 20}, {'price': 15}],
            'tax_rate': 0.1,
            'discount': 5
        }

        subtotal = sum(item['price'] for item in order['items'])
        assert subtotal == 45

        discount = order['discount']
        after_discount = subtotal - discount
        assert after_discount == 40

        tax = after_discount * order['tax_rate']
        assert tax == 4

        total = after_discount + tax
        assert total == 44

    def test_user_registration_workflow(self):
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'securepass123'
        }

        assert validate_email(user_data['email']) is True
        assert len(user_data['username']) > 0
        assert len(user_data['password']) >= 8

        stored_user = user_data.copy()
        assert stored_user['email'] == 'test@example.com'


class TestIntegrationDataProcessing:
    def test_batch_processing(self):
        numbers = [1, 2, 3, 4, 5]
        processed = []

        for num in numbers:
            result = multiply(num, 10)
            processed.append(result)

        assert processed == [10, 20, 30, 40, 50]

        total = sum(processed)
        assert total == 150

    def test_filter_and_transform(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        even_numbers = [x for x in data if x % 2 == 0]
        assert even_numbers == [2, 4, 6, 8, 10]

        doubled = [multiply(x, 2) for x in even_numbers]
        assert doubled == [4, 8, 12, 16, 20]
