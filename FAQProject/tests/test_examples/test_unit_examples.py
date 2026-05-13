import pytest


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


def validate_email(email):
    if not isinstance(email, str):
        return False
    if '@' not in email or '.' not in email:
        return False
    parts = email.split('@')
    if len(parts) != 2:
        return False
    local, domain = parts
    return len(local) > 0 and len(domain) > 1 and '.' in domain


def validate_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def reverse_string(text):
    return text[::-1]


def calculate_factorial(n):
    if n < 0:
        raise ValueError("负数没有阶乘")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)


class TestCalculatorFunctions:
    def test_add_positive_numbers(self):
        assert add(2, 3) == 5

    def test_add_negative_numbers(self):
        assert add(-1, -1) == -2

    def test_add_mixed_numbers(self):
        assert add(-5, 3) == -2

    def test_add_zero(self):
        assert add(0, 0) == 0

    def test_subtract(self):
        assert subtract(10, 3) == 7

    def test_subtract_negative_result(self):
        assert subtract(3, 10) == -7

    def test_multiply(self):
        assert multiply(4, 5) == 20

    def test_multiply_by_zero(self):
        assert multiply(100, 0) == 0

    def test_divide(self):
        assert divide(10, 2) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="除数不能为零"):
            divide(10, 0)

    def test_divide_negative(self):
        assert divide(-10, 2) == -5


class TestValidationFunctions:
    def test_valid_email(self):
        assert validate_email("test@example.com") is True

    def test_email_without_at(self):
        assert validate_email("testexample.com") is False

    def test_email_without_dot(self):
        assert validate_email("test@examplecom") is False

    def test_email_empty(self):
        assert validate_email("") is False

    def test_email_not_string(self):
        assert validate_email(123) is False

    def test_valid_number(self):
        assert validate_number("123") is True

    def test_valid_float(self):
        assert validate_number("123.45") is True

    def test_negative_number(self):
        assert validate_number("-123") is True

    def test_invalid_number(self):
        assert validate_number("abc") is False

    def test_mixed_string(self):
        assert validate_number("123abc") is False


class TestStringFunctions:
    def test_reverse_string(self):
        assert reverse_string("hello") == "olleh"

    def test_reverse_empty_string(self):
        assert reverse_string("") == ""

    def test_reverse_single_char(self):
        assert reverse_string("a") == "a"

    def test_reverse_palindrome(self):
        assert reverse_string("radar") == "radar"


class TestFactorialFunction:
    def test_factorial_zero(self):
        assert calculate_factorial(0) == 1

    def test_factorial_one(self):
        assert calculate_factorial(1) == 1

    def test_factorial_five(self):
        assert calculate_factorial(5) == 120

    def test_factorial_negative(self):
        with pytest.raises(ValueError, match="负数没有阶乘"):
            calculate_factorial(-5)
