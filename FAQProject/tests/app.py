from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({
        'message': '测试技术展示API',
        'endpoints': {
            'health': '/health',
            'calculator': '/api/calculator',
            'string_ops': '/api/string_ops',
            'data_validation': '/api/validate'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'test-demo-api'})


@app.route('/api/calculator', methods=['POST'])
def calculator():
    data = request.get_json()
    operation = data.get('operation')
    a = float(data.get('a', 0))
    b = float(data.get('b', 0))

    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else None
    }

    if operation not in operations:
        return jsonify({'error': '无效的操作'}), 400

    result = operations[operation](a, b)
    if result is None:
        return jsonify({'error': '除数不能为零'}), 400

    return jsonify({
        'operation': operation,
        'a': a,
        'b': b,
        'result': result
    })


@app.route('/api/string_ops', methods=['POST'])
def string_ops():
    data = request.get_json()
    text = data.get('text', '')
    operation = data.get('operation')

    operations = {
        'upper': str.upper,
        'lower': str.lower,
        'reverse': lambda x: x[::-1],
        'length': len
    }

    if operation not in operations:
        return jsonify({'error': '无效的操作'}), 400

    result = operations[operation](text)
    return jsonify({
        'operation': operation,
        'input': text,
        'result': result
    })


@app.route('/api/validate', methods=['POST'])
def validate():
    data = request.get_json()
    value = data.get('value')
    validation_type = data.get('type')

    validators = {
        'email': lambda x: '@' in str(x) and '.' in str(x),
        'number': lambda x: str(x).replace('.', '').replace('-', '').isdigit(),
        'positive': lambda x: float(x) > 0 if str(x).replace('.', '').replace('-', '').isdigit() else False,
        'string': lambda x: isinstance(x, str)
    }

    if validation_type not in validators:
        return jsonify({'error': '无效的验证类型'}), 400

    is_valid = validators[validation_type](value)
    return jsonify({
        'type': validation_type,
        'value': value,
        'is_valid': is_valid
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)