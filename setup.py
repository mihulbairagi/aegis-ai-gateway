from setuptools import setup, find_packages

setup(
    name="aegis-ai-gateway",
    version="1.3.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "numpy>=1.26.0",
        "pydantic>=2.7.0",
        "httpx>=0.27.0",
        "rich>=13.7.0",
        "fastembed>=0.4.0",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "aegis=app.cli:main",
        ],
    },
)
