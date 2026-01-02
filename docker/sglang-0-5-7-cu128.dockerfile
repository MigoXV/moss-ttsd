# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/sglang:0.5.7-cu128

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY wheels/ ./wheels/
RUN /app/.venv/bin/pip install accelerate>=1.9.0 && \
    /app/.venv/bin/pip cache purge && \
    rm -rf wheels

# 拷贝必要的文件以安装依赖
COPY pyproject.toml poetry.lock README.md ./
RUN mkdir -p src/moss_ttsd && \
    touch src/moss_ttsd/__init__.py && \
    poetry install --no-root

# 拷贝源代码文件
COPY . .

# 安装当前包
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 8000

# 默认入口
CMD ["poetry", "run", "python", "-m", "moss_ttsd.commands.app"]
