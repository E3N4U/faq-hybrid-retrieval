# Face Recognition API - Testing Project

## 项目简介

这是一个完整的人脸识别面试测试项目，展示了从基础算法到 DevOps 完整流程的测试技术。

## 技术栈

- **后端**: Flask + OpenCV
- **测试**: pytest + requests
- **容器化**: Docker + Docker Compose
- **CI/CD**: GitHub Actions

## 项目结构

```
face_recognition_project/
├── app/                      # 应用代码
│   ├── __init__.py
│   ├── face_recognition.py  # 人脸识别核心算法
│   └── api.py               # Flask API 接口
├── tests/                    # 测试代码
│   ├── conftest.py          # pytest fixtures
│   ├── test_unit.py         # 单元测试
│   ├── test_api.py          # API 测试
│   └── test_ui.py           # UI 测试
├── ui/                      # 前端页面
│   └── index.html           # 人脸识别上传界面
├── docker/                  # Docker 配置
│   ├── Dockerfile           # Docker 镜像配置
│   └── docker-compose.yml   # Docker Compose 配置
├── ci/                      # CI/CD 配置
│   └── github-actions.yml   # GitHub Actions 配置
├── requirements.txt         # Python 依赖
└── pytest.ini              # pytest 配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
pytest tests/ -v
```

### 3. 启动 API

```bash
python app/api.py
```

### 4. Docker 部署

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## API 接口

- `GET /api/health` - 健康检查
- `POST /api/detect/base64` - Base64 图片检测
- `POST /api/detect/file` - 文件上传检测
- `POST /api/detect/url` - URL 图片检测

## 测试覆盖

- ✅ 单元测试 - 人脸识别算法
- ✅ API 测试 - HTTP 接口
- ✅ UI 测试 - 前端交互
- ✅ 集成测试 - 多组件协同
- ✅ Docker 部署测试
